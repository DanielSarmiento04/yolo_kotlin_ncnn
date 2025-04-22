#include <jni.h>
#include <string>
#include <vector>
#include <algorithm> // For std::max, std::min, std::sort
#include <cmath>     // Use cmath instead of math.h for C++ style, needed for expf
#include <numeric>   // For std::iota
#include <chrono>    // For timing
#include <thread>    // For std::thread::hardware_concurrency

// Android Logging
#include <android/log.h>
#define LOG_TAG "NCNN_YOLOv11_Native" // More specific tag
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__) // Debug logs

// Asset Manager (for loading models from assets)
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

// NCNN Headers
#include "ncnn/net.h"
#include "ncnn/gpu.h"
#include "ncnn/layer.h" // Required for custom layers if any
#include "ncnn/mat.h"   // For ncnn::Mat operations
#include "ncnn/cpu.h"   // For ncnn misc functions like yuv420sp2rgb, resize_bilinear, copy_make_border

// Global NCNN Net instance and state variables
static ncnn::Net yoloNet;
static ncnn::Mutex yoloNetLock; // For thread safety if calling detect from multiple threads
static bool ncnnInitialized = false;
static bool modelLoaded = false;
static bool useGPU = false;     // Whether Vulkan will be used
static int gpuDeviceIndex = -1; // Store selected GPU device index
static bool gpuInstanceCreated = false; // Track if create_gpu_instance was called

// YOLOv11 constants (ADJUST THESE BASED ON YOUR SPECIFIC YOLOv11 MODEL)
const int YOLOV11_INPUT_WIDTH = 640;
const int YOLOV11_INPUT_HEIGHT = 640;
// Use thresholds consistent with Kotlin side or typical values
const float NMS_THRESHOLD = 0.45f;        // Non-Maximum Suppression threshold
// *** RESTORE ORIGINAL CONFIDENCE THRESHOLD ***
const float CONFIDENCE_THRESHOLD = 0.25f;
const float CLASS_SCORE_THRESHOLD = 0.25f; // Restore matching threshold
const int NUM_CLASSES = 84;               // Number of classes the model can detect (updated to 84)

// Structure to hold detection results internally before NMS
struct Object
{
    float x;    // Top-left corner x (relative to original image width)
    float y;    // Top-left corner y (relative to original image height)
    float w;    // Width (relative to original image width)
    float h;    // Height (relative to original image height)
    int label;  // Class index
    float prob; // Confidence score
};

// Helper function for Non-Maximum Suppression (NMS) - Intersection over Union
// Calculates the intersection area between two bounding boxes.
static inline float intersection_area(const Object &a, const Object &b)
{
    // Determine the coordinates of the intersection rectangle
    float inter_x1 = std::max(a.x, b.x);
    float inter_y1 = std::max(a.y, b.y);
    float inter_x2 = std::min(a.x + a.w, b.x + b.w);
    float inter_y2 = std::min(a.y + a.h, b.y + b.h);

    // Calculate the width and height of the intersection rectangle
    // Ensure width and height are non-negative
    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);

    // Calculate the intersection area
    return inter_w * inter_h;
}

// NMS function - Filters overlapping bounding boxes based on IoU.
// Assumes input `objects` are pre-sorted by confidence score in descending order.
static void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();
    const size_t n = objects.size(); // Use size_t for sizes
    if (n == 0)
        return;

    // Pre-calculate areas of all bounding boxes
    std::vector<float> areas(n);
    for (size_t i = 0; i < n; ++i) // Use size_t for loop index
    {
        areas[i] = objects[i].w * objects[i].h;
        // Ensure area is non-negative, although width/height should already be positive
        if (areas[i] < 0)
            areas[i] = 0;
    }

    // Keep track of boxes that are suppressed (true) or kept (false)
    std::vector<bool> suppressed(n, false);

    // Iterate through the sorted boxes
    for (size_t i = 0; i < n; ++i)
    {
        // Skip if this box has already been suppressed by a higher-confidence box
        if (suppressed[i])
        {
            continue;
        }

        // This box is not suppressed, so pick it
        picked.push_back(static_cast<int>(i)); // Add index to the list of picked boxes
        const Object &a = objects[i];          // Reference to the current box

        // Compare the current box 'a' with all subsequent boxes 'b'
        for (size_t j = i + 1; j < n; ++j)
        {
            // Skip if box 'j' has already been suppressed
            if (suppressed[j])
            {
                continue;
            }
            const Object &b = objects[j]; // Reference to the box being compared

            // Calculate Intersection over Union (IoU)
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[j] - inter_area;

            // Avoid division by zero or very small numbers
            if (union_area > 1e-6f) // Use a small epsilon (float literal)
            {
                float iou = inter_area / union_area;

                // If IoU exceeds the threshold, suppress box 'j'
                if (iou > nms_threshold)
                {
                    suppressed[j] = true;
                }
            }
            // Handle edge case where union_area is near zero (e.g., identical boxes)
            // If intersection is significant, suppress. This depends on desired behavior for identical boxes.
            else if (inter_area > 1e-6f)
            {
                suppressed[j] = true;
            }
        }
    }
}

// Sigmoid activation function
static inline float sigmoid(float x)
{
    // Use expf for float version of exp
    return 1.0f / (1.0f + expf(-x));
}

extern "C"
{

    // JNI Function: Initialize NCNN environment and check for GPU support.
    JNIEXPORT jboolean JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_initNative(JNIEnv *env, jobject /* this */, jobject assetManager)
    {
        ncnn::MutexLockGuard guard(yoloNetLock);

        if (ncnnInitialized)
        {
            LOGI("NCNN already initialized.");
            return JNI_TRUE;
        }
        LOGI("Initializing NCNN for YOLOv11...");
        auto init_start_time = std::chrono::high_resolution_clock::now();

        // Get AAssetManager - needed for model loading later
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        if (mgr == nullptr)
        {
            LOGE("Failed to get AAssetManager from Java context.");
        }

        // Check Vulkan support
        int gpu_count = ncnn::get_gpu_count();
        if (gpu_count > 0)
        {
            gpuDeviceIndex = ncnn::get_default_gpu_index();
            useGPU = true;
            LOGI("Vulkan GPU detected. Count: %d. Selected device index: %d. Enabling GPU acceleration.", gpu_count, gpuDeviceIndex);

            // Create GPU instance *once* during initialization
            if (!gpuInstanceCreated)
            {
                ncnn::create_gpu_instance();
                gpuInstanceCreated = true; // Mark as created
                LOGI("NCNN Vulkan GPU instance created.");
            }
            else
            {
                LOGI("NCNN Vulkan GPU instance already exists.");
            }
        }
        else
        {
            useGPU = false;
            gpuDeviceIndex = -1;
            LOGW("No Vulkan capable GPU detected or NCNN not built with Vulkan support. Using CPU.");
        }

        // Configure NCNN Net options
        yoloNet.opt = ncnn::Option();
        yoloNet.opt.lightmode = true;
        yoloNet.opt.num_threads = std::min(4, (int)std::thread::hardware_concurrency());
        yoloNet.opt.use_packing_layout = true; // Essential for performance
        yoloNet.opt.use_fp16_packed = useGPU;  // Use FP16 packed on GPU
        yoloNet.opt.use_fp16_storage = useGPU; // Store weights in FP16 on GPU
        yoloNet.opt.use_fp16_arithmetic = false; // Usually false is safer/faster unless tested
        yoloNet.opt.use_vulkan_compute = useGPU; // Enable Vulkan

        if (useGPU)
        {
            yoloNet.set_vulkan_device(gpuDeviceIndex);
            LOGI("NCNN Net configured to use Vulkan device %d.", gpuDeviceIndex);
        }
        else
        {
            LOGI("NCNN Net configured to use CPU with %d threads.", yoloNet.opt.num_threads);
        }

        ncnnInitialized = true;
        modelLoaded = false;
        auto init_end_time = std::chrono::high_resolution_clock::now();
        auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end_time - init_start_time);
        LOGI("NCNN initialization complete (took %lld ms). Vulkan enabled: %s", init_duration.count(), useGPU ? "true" : "false");
        return JNI_TRUE;
    }

    // JNI Function: Load the YOLO model parameters and binary weights.
    JNIEXPORT jboolean JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_loadModelNative(JNIEnv *env, jobject /* this */, jobject assetManager)
    {
        ncnn::MutexLockGuard guard(yoloNetLock);

        if (!ncnnInitialized)
        {
            LOGE("NCNN not initialized. Call initNative() before loading model.");
            return JNI_FALSE;
        }
        if (modelLoaded)
        {
            LOGI("Model already loaded.");
            return JNI_TRUE;
        }
        LOGI("Loading YOLOv11 model...");
        auto load_start_time = std::chrono::high_resolution_clock::now();

        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        if (mgr == nullptr)
        {
            LOGE("Failed to get AAssetManager from Java context for model loading.");
            return JNI_FALSE;
        }

        // *** ADJUST THESE FILENAMES TO MATCH YOUR YOLOv11 NCNN MODEL FILES ***
        const char *param_filename = "yolov11.param";
        const char *bin_filename = "yolov11.bin";

        // Load model parameters
        int ret_param = yoloNet.load_param(mgr, param_filename);
        if (ret_param != 0)
        {
            LOGE("Failed to load model param file: %s (Error code: %d). Check assets.", param_filename, ret_param);
            return JNI_FALSE;
        }
        LOGD("Loaded model param: %s", param_filename);

        // Load model weights
        int ret_bin = yoloNet.load_model(mgr, bin_filename);
        if (ret_bin != 0)
        {
            LOGE("Failed to load model bin file: %s (Error code: %d). Check assets.", bin_filename, ret_bin);
            yoloNet.clear(); // Clear partially loaded state
            return JNI_FALSE;
        }
        LOGD("Loaded model bin: %s", bin_filename);

        modelLoaded = true;
        auto load_end_time = std::chrono::high_resolution_clock::now();
        auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);
        LOGI("YOLOv11 model loading complete (took %lld ms).", load_duration.count());
        return JNI_TRUE;
    }

    // JNI Function: Check if Vulkan (GPU) is initialized and being used.
    JNIEXPORT jboolean JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_hasVulkan(JNIEnv *env, jobject /* this */)
    {
        // Reading bools is atomic enough, lock not strictly needed here
        return (ncnnInitialized && useGPU) ? JNI_TRUE : JNI_FALSE;
    }

    // JNI Function: Perform object detection on YUV image data provided via ByteBuffers.
    JNIEXPORT jfloatArray JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_detectNative(
        JNIEnv *env, jobject /* this */,
        jobject yBuffer, jobject uBuffer, jobject vBuffer, // Pass all three
        jint yStride, jint uvStride, jint uvPixelStride,
        jint imageWidth, jint imageHeight)
    {
        // --- Pre-checks ---
        if (!ncnnInitialized || !modelLoaded)
        {
            LOGE("Detection failed: NCNN not initialized or model not loaded.");
            return nullptr;
        }
        // Check required buffers (Y and V are essential for NV21 conversion)
        if (yBuffer == nullptr || vBuffer == nullptr || imageWidth <= 0 || imageHeight <= 0 || yStride <= 0 || uvStride <= 0)
        {
            LOGE("Detection failed: Invalid YUV input data provided (Y/V buffer null or invalid dims/strides/pixelStride).");
            return nullptr;
        }
        // Warn if pixel stride isn't 2, as ncnn::Mat::from_pixels_yuv420sp_resize assumes it.
        if (uvPixelStride != 2)
        {
            LOGW("Received uvPixelStride = %d. ncnn::Mat::from_pixels_yuv420sp_resize assumes 2. Conversion might be incorrect.", uvPixelStride);
        }

        // --- Overall Timing Start ---
        auto total_start_time = std::chrono::high_resolution_clock::now();

        // 1. Get direct buffer access to YUV planes
        unsigned char *y_pixel_data = (unsigned char *)env->GetDirectBufferAddress(yBuffer);
        unsigned char *vu_pixel_data = (unsigned char *)env->GetDirectBufferAddress(vBuffer); // For NV21, V buffer points to start of VU plane

        if (y_pixel_data == nullptr || vu_pixel_data == nullptr)
        {
            LOGE("Failed to get direct buffer address for Y or VU (V) planes.");
            return nullptr;
        }

        // --- Preprocessing Timing Start ---
        auto preprocess_start_time = std::chrono::high_resolution_clock::now();

        // 2. Preprocessing: Manual YUV420SP (NV21) -> RGB -> Resize -> Pad -> Normalize

        // Calculate scaling factor and padding for letterboxing/pillarboxing
        float scale_x = (float)YOLOV11_INPUT_WIDTH / (float)imageWidth;
        float scale_y = (float)YOLOV11_INPUT_HEIGHT / (float)imageHeight;
        float scale = std::min(scale_x, scale_y);

        int scaled_w = static_cast<int>(imageWidth * scale);
        int scaled_h = static_cast<int>(imageHeight * scale);

        // Calculate padding offsets
        // Ensure pads are non-negative integers
        int pad_top = std::max(0, (YOLOV11_INPUT_HEIGHT - scaled_h) / 2);
        int pad_bottom = std::max(0, YOLOV11_INPUT_HEIGHT - scaled_h - pad_top);
        int pad_left = std::max(0, (YOLOV11_INPUT_WIDTH - scaled_w) / 2);
        int pad_right = std::max(0, YOLOV11_INPUT_WIDTH - scaled_w - pad_left);

        ncnn::Mat input_img; // This will hold the final preprocessed image

        { // Scope for temporary buffers and mats
            // Allocate temporary buffer for the full original RGB image
            std::vector<unsigned char> rgb_buffer(imageWidth * imageHeight * 3);

            // --- Create a contiguous YUV buffer ---
            // ncnn::yuv420sp2rgb expects Y plane followed immediately by VU plane.
            // Size = Y_size + VU_size = (w*h) + (w*h/2) = w*h*3/2
            size_t y_size = static_cast<size_t>(imageWidth) * imageHeight; // Use imageWidth, function assumes no stride padding
            size_t vu_size = y_size / 2; // VU plane size
            std::vector<unsigned char> yuv_buffer(y_size + vu_size);

            // Copy Y plane data.
            // WARNING: This assumes yStride == imageWidth. If not, this copy is incorrect.
            // A row-by-row copy would be needed if yStride != imageWidth.
            memcpy(yuv_buffer.data(), y_pixel_data, y_size);

            // Copy VU plane data immediately after Y data.
            // WARNING: This assumes uvStride handles the interleaved VU data correctly
            // and effectively uvStride/2 == imageWidth/2 for the number of V/U pairs per row.
            // A more robust copy would handle uvPixelStride and uvStride explicitly row by row.
            // For simplicity matching yuv420sp2rgb's expectation, we copy vu_size bytes.
            memcpy(yuv_buffer.data() + y_size, vu_pixel_data, vu_size);
            // --- End contiguous YUV buffer creation ---


            // Convert YUV420SP (NV21 assumed) to RGB using the contiguous buffer
            // Pass the pointer to the start of our combined buffer.
            ncnn::yuv420sp2rgb(yuv_buffer.data(), imageWidth, imageHeight, rgb_buffer.data());

            // Create ncnn::Mat from the original size RGB data
            ncnn::Mat rgb_mat = ncnn::Mat::from_pixels(rgb_buffer.data(), ncnn::Mat::PIXEL_RGB, imageWidth, imageHeight);
            if (rgb_mat.empty())
            {
                LOGE("Failed to create ncnn::Mat from RGB buffer.");
                return nullptr;
            }

            // Resize the RGB Mat to the scaled dimensions (maintaining aspect ratio)
            ncnn::Mat resized_mat;
            ncnn::resize_bilinear(rgb_mat, resized_mat, scaled_w, scaled_h);
            if (resized_mat.empty())
            {
                LOGE("Failed to resize RGB ncnn::Mat.");
                return nullptr;
            }

            // Pad the resized Mat to the final target input size
            ncnn::copy_make_border(resized_mat, input_img, pad_top, pad_bottom, pad_left, pad_right, ncnn::BORDER_CONSTANT, 114.f);
            if (input_img.empty())
            {
                LOGE("Failed to pad resized ncnn::Mat.");
                return nullptr;
            }
        } // End scope for temporary buffers/mats

        // Normalize the final padded image (applied to the RGB Mat)
        const float mean_vals[3] = {0.f, 0.f, 0.f};
        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        input_img.substract_mean_normalize(mean_vals, norm_vals);

        // --- Preprocessing Timing End ---
        auto preprocess_end_time = std::chrono::high_resolution_clock::now();
        auto preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end_time - preprocess_start_time);

        // 3. NCNN Inference
        ncnn::Mat out;
        const char *input_name = "in0";   // Verify this matches your .param file
        const char *output_name = "out0"; // Verify this matches your .param file
        std::chrono::microseconds inference_duration(0);

        { // Scope for extractor and mutex lock
            ncnn::MutexLockGuard guard(yoloNetLock);
            if (!modelLoaded)
            {
                LOGE("Detection failed inside lock: Model not loaded.");
                return nullptr;
            }
            ncnn::Extractor ex = yoloNet.create_extractor();

            auto inference_start_time = std::chrono::high_resolution_clock::now();

            int input_ret = ex.input(input_name, input_img);
            if (input_ret != 0)
            {
                LOGE("Failed to set input tensor '%s'. Error: %d.", input_name, input_ret);
                return nullptr;
            }

            int extract_ret = ex.extract(output_name, out);
            if (extract_ret != 0)
            {
                LOGE("Failed to extract output tensor '%s'. Error: %d.", output_name, extract_ret);
                return nullptr;
            }

            auto inference_end_time = std::chrono::high_resolution_clock::now();
            inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end_time - inference_start_time);
        } // Mutex released, extractor destroyed

        // --- Postprocessing Timing Start ---
        auto postprocess_start_time = std::chrono::high_resolution_clock::now();

        std::vector<Object> proposals;
        std::vector<Object> raw_objects; // Keep storing raw objects before NMS

        // --- Adjust Dimension Check ---
        // Allow Dims=2 (common format [num_proposals, num_features])
        // Also check C dimension if Dims=3 (less common now)
        if (out.dims != 2 && (out.dims != 3 || out.c != 1))
        {
            LOGE("Unexpected output tensor shape. Dims=%d, W=%d, H=%d, C=%d. Expected Dims=2 or Dims=3 with C=1.", out.dims, out.w, out.h, out.c);
            return nullptr;
        }

        int num_proposals;
        int num_features;

        if (out.dims == 2)
        {
            num_proposals = out.w;
            num_features = out.h;
            LOGD("Output tensor dims=2. num_proposals(W)=%d, num_features(H)=%d", num_proposals, num_features);
        }
        else
        { // out.dims == 3 && out.c == 1
            num_proposals = out.w;
            num_features = out.h;
            LOGD("Output tensor dims=3, C=1. num_proposals(W)=%d, num_features(H)=%d", num_proposals, num_features);
        }
        // --- End Dimension Check Adjustment ---

        // *** UPDATED EXPECTED FEATURES CALCULATION ***
        int expected_features = 4 + NUM_CLASSES; // cx, cy, w, h, class_scores...

        LOGD("Checking features: Actual num_features = %d, Expected features = %d (4+%d)", num_features, expected_features, NUM_CLASSES);

        if (num_features != expected_features)
        {
            LOGE("Output feature count mismatch! Expected: %d (4 bbox + %d classes), Got: %d.", expected_features, NUM_CLASSES, num_features);
            return nullptr; // Return null on mismatch
        }

        if (num_proposals <= 0)
        {
            LOGW("No proposals generated by the model (num_proposals = %d).", num_proposals);
            // Return an array indicating zero detections
            jfloatArray emptyResult = env->NewFloatArray(1);
            if (emptyResult)
            {
                float zero = 0.0f;
                env->SetFloatArrayRegion(emptyResult, 0, 1, &zero);
            }
            else
            {
                LOGE("Failed to allocate empty result array.");
            }
            return emptyResult; // Return empty array instead of null
        }

        raw_objects.reserve(num_proposals / 10); // Reserve space

        const float *data = (const float *)out.data;

        // *** UPDATED POST-PROCESSING LOOP with RAW LOGIT LOGGING ***
        int proposal_log_counter = 0; // Counter for limiting proposal logs
        const int MAX_PROPOSAL_LOGS = 5; // Log logits for first few proposals

        for (int i = 0; i < num_proposals; ++i)
        {
            // Data layout assumed: [cx, cy, w, h, class0_logit, class1_logit, ...]
            const float *proposal_data = data + i * num_features;

            // Find the class with the highest score for this proposal
            int best_class_idx = -1;
            float max_class_score_prob = -1.0f; // Store max probability

            const float *class_scores_ptr = proposal_data + 4;

            for (int c = 0; c < NUM_CLASSES; ++c)
            {
                float class_score_logit = class_scores_ptr[c];

                // *** ADDED RAW LOGIT LOGGING (Limited) ***
                if (proposal_log_counter < MAX_PROPOSAL_LOGS && c < 5)
                { // Log first 5 classes for first 5 proposals
                    LOGD("Proposal %d, Class %d: Raw Logit = %.4f", i, c, class_score_logit);
                }
                else if (proposal_log_counter < MAX_PROPOSAL_LOGS && class_score_logit > 10.0f)
                { // Log if logit is unusually high
                    LOGD("Proposal %d, Class %d: High Raw Logit = %.4f", i, c, class_score_logit);
                }
                // *** END RAW LOGIT LOGGING ***

                float class_score_prob = sigmoid(class_score_logit);

                if (class_score_prob > max_class_score_prob)
                {
                    max_class_score_prob = class_score_prob;
                    best_class_idx = c;
                }
            }
            // Increment proposal log counter after processing all classes for one proposal
            if (i < MAX_PROPOSAL_LOGS)
            {
                proposal_log_counter++;
            }

            // Check if the highest class score meets the (restored) confidence threshold
            if (max_class_score_prob >= CONFIDENCE_THRESHOLD)
            {
                // Extract bounding box coordinates (center x, center y, width, height)
                float cx_net = proposal_data[0];
                float cy_net = proposal_data[1];
                float w_net = proposal_data[2];
                float h_net = proposal_data[3];

                // ... Coordinate conversion and clamping logic remains the same ...
                float x1_net = cx_net - w_net / 2.0f;
                float y1_net = cy_net - h_net / 2.0f;

                float x1_orig = (x1_net - pad_left) / scale;
                float y1_orig = (y1_net - pad_top) / scale;
                float w_orig = w_net / scale;
                float h_orig = h_net / scale;

                x1_orig = std::max(0.0f, std::min((float)imageWidth - 1.0f, x1_orig));
                y1_orig = std::max(0.0f, std::min((float)imageHeight - 1.0f, y1_orig));
                float x2_orig = std::min((float)imageWidth - 1.0f, x1_orig + w_orig);
                float y2_orig = std::min((float)imageHeight - 1.0f, y1_orig + h_orig);

                w_orig = x2_orig - x1_orig;
                h_orig = y2_orig - y1_orig;

                if (w_orig > 0 && h_orig > 0)
                {
                    Object obj;
                    obj.x = x1_orig;
                    obj.y = y1_orig;
                    obj.w = w_orig;
                    obj.h = h_orig;
                    obj.label = best_class_idx;
                    obj.prob = max_class_score_prob;
                    raw_objects.push_back(obj); // Add to raw_objects
                }
            }
        }

        // Sort objects by confidence score (descending)
        std::sort(raw_objects.begin(), raw_objects.end(), [](const Object &a, const Object &b)
                  { return a.prob > b.prob; });

        // *** RESTORED NMS ***
        std::vector<int> picked_indices;
        nms_sorted_bboxes(raw_objects, picked_indices, NMS_THRESHOLD);

        // Collect final proposals after NMS
        proposals.reserve(picked_indices.size());
        for (int index : picked_indices)
        {
            if (index >= 0 && static_cast<size_t>(index) < raw_objects.size())
            {
                proposals.push_back(raw_objects[index]);
            }
            else
            {
                LOGW("Invalid index %d from NMS, skipping.", index);
            }
        }
        // *** END RESTORED NMS ***

        // --- Postprocessing Timing End ---
        auto postprocess_end_time = std::chrono::high_resolution_clock::now();
        auto postprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end_time - postprocess_start_time);

        // 5. Format results for Java/Kotlin
        int final_count = proposals.size(); // Use size of proposals
        int result_elements = 1 + final_count * 6;
        jfloatArray resultArray = env->NewFloatArray(result_elements);
        if (resultArray == nullptr)
        {
            LOGE("Failed to allocate float array for JNI results (size %d).", result_elements);
            return nullptr;
        }

        std::vector<float> resultData(result_elements);
        resultData[0] = static_cast<float>(final_count);

        for (int i = 0; i < final_count; ++i)
        {
            const Object &obj = proposals[i]; // Use proposals
            int offset = 1 + i * 6;
            resultData[offset + 0] = obj.x;
            resultData[offset + 1] = obj.y;
            resultData[offset + 2] = obj.w;
            resultData[offset + 3] = obj.h;
            resultData[offset + 4] = static_cast<float>(obj.label);
            resultData[offset + 5] = obj.prob;
        }

        env->SetFloatArrayRegion(resultArray, 0, result_elements, resultData.data());

        // --- Overall Timing End & Logging ---
        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);

        static int frame_counter = 0;
        static const int LOG_INTERVAL = 30;
        if (++frame_counter % LOG_INTERVAL == 0)
        {
            LOGI("Detection Timing (us): Total=%lld | Preproc=%lld | Infer=%lld | Postproc=%lld | Objects=%d",
                 total_duration_us.count(), preprocess_duration.count(), inference_duration.count(), postprocess_duration.count(), final_count);
        }

        return resultArray;
    }

    // JNI Function: Release NCNN resources, including network and GPU context.
    JNIEXPORT void JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_releaseNative(JNIEnv *env, jobject /* this */)
    {
        LOGI("Releasing NCNN resources...");
        bool wasUsingGpu;
        {
            ncnn::MutexLockGuard guard(yoloNetLock);
            wasUsingGpu = useGPU;

            if (modelLoaded)
            {
                yoloNet.clear();
                LOGD("NCNN Net cleared.");
                modelLoaded = false;
            }
            else
            {
                LOGD("NCNN Net was not loaded, skipping clear.");
            }

            ncnnInitialized = false;
            useGPU = false;
            gpuDeviceIndex = -1;
        }

        if (wasUsingGpu && gpuInstanceCreated)
        {
            ncnn::destroy_gpu_instance();
            gpuInstanceCreated = false;
            LOGI("Vulkan GPU instance destroyed.");
        }
        else if (wasUsingGpu && !gpuInstanceCreated)
        {
            LOGW("Attempting to destroy GPU instance, but it wasn't marked as created.");
        }
        else
        {
            LOGI("No Vulkan GPU instance to destroy (was using CPU or instance not created).");
        }

        LOGI("NCNN resources released and state reset.");
    }

} // extern "C"
