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
// CONFIDENCE_THRESHOLD now applies directly to the max class score
const float CONFIDENCE_THRESHOLD = 0.25f;
// CLASS_SCORE_THRESHOLD might be redundant if CONFIDENCE_THRESHOLD serves the purpose,
// but keep it for potential future use or set it equal to CONFIDENCE_THRESHOLD.
const float CLASS_SCORE_THRESHOLD = 0.25f; // Can be adjusted or removed if redundant
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

// Helper function to clamp value within 0-255 range
static inline unsigned char clamp_u8(int value)
{
    return static_cast<unsigned char>(std::max(0, std::min(255, value)));
}

// Manual YUV420SP (NV21) to RGB conversion
// Assumes NV21 format: Y plane followed by an interleaved V/U plane (VUVUVU...)
// y_plane: Pointer to the start of the Y data
// vu_plane: Pointer to the start of the interleaved V/U data (points to the first V)
// rgb_output: Pointer to the output buffer (must be pre-allocated: width * height * 3 bytes)
// width, height: Dimensions of the image
// y_stride: Row stride of the Y plane
// uv_stride: Row stride of the V/U plane
// uv_pixel_stride: Pixel stride within the V/U plane (should be 2 for NV21)
static void nv21_to_rgb(const unsigned char *y_plane, const unsigned char *vu_plane,
                        unsigned char *rgb_output,
                        int width, int height, int y_stride, int uv_stride, int uv_pixel_stride)
{
    // Basic check for NV21 compatibility
    if (uv_pixel_stride != 2)
    {
        LOGE("nv21_to_rgb: Expected uv_pixel_stride of 2 for NV21 format, got %d. Filling output with black.", uv_pixel_stride);
        std::fill(rgb_output, rgb_output + width * height * 3, 0);
        return;
    }

    for (int j = 0; j < height; ++j)
    {
        const unsigned char *y_row = y_plane + j * y_stride;
        // UV plane row index depends on Y row index (UV height is half of Y height)
        // UV data for Y rows j and j+1 is located at UV row j/2
        const unsigned char *uv_row = vu_plane + (j / 2) * uv_stride;

        for (int i = 0; i < width; ++i)
        {
            // Y value for pixel (i, j)
            const int y_value = y_row[i];

            // Calculate index for V and U values in the interleaved UV plane
            // For NV21 (VUVUVU): V is at uv_row[ (i/2) * 2 ], U is at uv_row[ (i/2) * 2 + 1 ]
            const int uv_index = (i / 2) * uv_pixel_stride; // uv_pixel_stride is 2
            const int v_value = uv_row[uv_index] - 128;     // V value for the 2x2 block
            const int u_value = uv_row[uv_index + 1] - 128; // U value for the 2x2 block

            // Calculate RGB values using standard conversion formula
            // These formulas can vary slightly.
            int r = y_value + (int)(1.402f * v_value);
            int g = y_value - (int)(0.344f * u_value + 0.714f * v_value);
            int b = y_value + (int)(1.772f * u_value);

            // Clamp values to [0, 255] and store in the output buffer (RGB order)
            int output_index = (j * width + i) * 3;
            rgb_output[output_index + 0] = clamp_u8(r); // R
            rgb_output[output_index + 1] = clamp_u8(g); // G
            rgb_output[output_index + 2] = clamp_u8(b); // B
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
        if (yBuffer == nullptr || vBuffer == nullptr || imageWidth <= 0 || imageHeight <= 0 || yStride <= 0 || uvStride <= 0 || uvPixelStride <= 0)
        {
            LOGE("Detection failed: Invalid YUV input data provided (Y/V buffer null or invalid dims/strides/pixelStride).");
            return nullptr;
        }
        // Warn if pixel stride isn't 2, as nv21_to_rgb expects it.
        if (uvPixelStride != 2)
        {
            LOGW("Received uvPixelStride = %d. nv21_to_rgb expects 2. Conversion might be incorrect.", uvPixelStride);
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

        // 2. Preprocessing: Manually convert YUV (NV21) to RGB, then create ncnn::Mat
        // Allocate buffer for RGB data
        std::vector<unsigned char> rgb_buffer(imageWidth * imageHeight * 3);

        // Perform manual conversion
        nv21_to_rgb(y_pixel_data, vu_pixel_data, rgb_buffer.data(),
                    imageWidth, imageHeight, yStride, uvStride, uvPixelStride);

        // Create ncnn::Mat from the converted RGB buffer
        // Use from_pixels_resize to handle potential resizing and letterboxing
        ncnn::Mat input_img = ncnn::Mat::from_pixels_resize(
            rgb_buffer.data(), ncnn::Mat::PIXEL_RGB,  // Input is now RGB
            imageWidth, imageHeight,                  // Original dimensions
            YOLOV11_INPUT_WIDTH, YOLOV11_INPUT_HEIGHT // Target model dimensions
        );
        // rgb_buffer goes out of scope here and is automatically deallocated

        if (input_img.empty())
        {
            LOGE("Failed to create or resize ncnn::Mat from manually converted RGB pixels.");
            return nullptr;
        }

        // Normalize the image (applied to the RGB Mat)
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

        if (out.dims == 2) {
            num_proposals = out.w;
            num_features = out.h;
            LOGD("Output tensor dims=2. num_proposals(W)=%d, num_features(H)=%d", num_proposals, num_features);
        } else { // out.dims == 3 && out.c == 1
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
            } else {
                 LOGE("Failed to allocate empty result array.");
            }
            return emptyResult; // Return empty array instead of null
        }

        float scale_x = (float)YOLOV11_INPUT_WIDTH / (float)imageWidth;
        float scale_y = (float)YOLOV11_INPUT_HEIGHT / (float)imageHeight;
        float scale = std::min(scale_x, scale_y);

        float pad_left = (YOLOV11_INPUT_WIDTH - imageWidth * scale) / 2.0f;
        float pad_top = (YOLOV11_INPUT_HEIGHT - imageHeight * scale) / 2.0f;

        std::vector<Object> raw_objects;
        raw_objects.reserve(num_proposals / 10);

        const float *data = (const float *)out.data;

        // *** UPDATED POST-PROCESSING LOOP ***
        for (int i = 0; i < num_proposals; ++i)
        {
            // Data layout assumed: [cx, cy, w, h, class0_logit, class1_logit, ...]
            const float *proposal_data = data + i * num_features;

            // Find the class with the highest score for this proposal
            int best_class_idx = -1;
            float max_class_score_prob = -1.0f; // Store max probability

            // Class scores start at index 4 in this format
            const float *class_scores_ptr = proposal_data + 4;

            for (int c = 0; c < NUM_CLASSES; ++c)
            {
                // Apply sigmoid to class score logit
                float class_score_logit = class_scores_ptr[c];
                float class_score_prob = sigmoid(class_score_logit);

                if (class_score_prob > max_class_score_prob)
                {
                    max_class_score_prob = class_score_prob;
                    best_class_idx = c;
                }
            }

            // Check if the highest class score meets the confidence threshold
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
                    // Final probability is the max class score probability
                    obj.prob = max_class_score_prob;
                    raw_objects.push_back(obj);
                }
            }
        }

        // Sort objects by confidence score (descending)
        std::sort(raw_objects.begin(), raw_objects.end(), [](const Object &a, const Object &b)
                  { return a.prob > b.prob; });

        // Apply Non-Maximum Suppression (NMS)
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

        // --- Postprocessing Timing End ---
        auto postprocess_end_time = std::chrono::high_resolution_clock::now();
        auto postprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end_time - postprocess_start_time);

        // 5. Format results for Java/Kotlin
        int final_count = proposals.size();
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
            const Object &obj = proposals[i];
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
