#include <jni.h>
#include <string>
#include <vector>
#include <algorithm> // For std::max, std::min, std::sort
#include <math.h>    // For expf, roundf, floorf
#include <numeric>   // For std::iota
#include <chrono>    // For timing

// Android Logging
#include <android/log.h>
#define LOG_TAG "NCNN_Native_YOLOv11"
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
static bool useGPU = false; // Whether Vulkan will be used

// YOLOv11 constants (ADJUST THESE BASED ON YOUR SPECIFIC YOLOv11 MODEL)
const int YOLOV11_INPUT_WIDTH = 640;
const int YOLOV11_INPUT_HEIGHT = 640;
// Use thresholds consistent with Kotlin side or typical values
const float NMS_THRESHOLD = 0.45f;       // Adjusted NMS threshold (common value)
const float CONFIDENCE_THRESHOLD = 0.25f; // Adjusted confidence threshold (common value)
const int NUM_CLASSES = 84;              // UPDATED: Match model.yml (0-83 -> 84 classes)

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
static inline float intersection_area(const Object &a, const Object &b)
{
    float inter_x1 = std::max(a.x, b.x);
    float inter_y1 = std::max(a.y, b.y);
    float inter_x2 = std::min(a.x + a.w, b.x + b.w);
    float inter_y2 = std::min(a.y + a.h, b.y + b.h);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);

    return inter_w * inter_h;
}

// NMS function - assumes input `objects` are pre-sorted by confidence
static void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();
    const int n = objects.size();
    if (n == 0)
        return;

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].w * objects[i].h;
    }

    std::vector<bool> suppressed(n, false); // Keep track of suppressed boxes

    for (int i = 0; i < n; i++)
    {
        if (suppressed[i])
        {
            continue; // Skip if already suppressed
        }
        picked.push_back(i); // Pick the current box
        const Object &a = objects[i];

        for (int j = i + 1; j < n; j++)
        {
            if (suppressed[j])
            {
                continue; // Skip if already suppressed
            }
            const Object &b = objects[j];

            // Intersection over Union (IoU)
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[j] - inter_area;

            // Avoid division by zero or near-zero
            if (union_area > 1e-6) // Use a small epsilon
            {
                float iou = inter_area / union_area;
                if (iou > nms_threshold)
                {
                    suppressed[j] = true; // Suppress box j
                }
            }
        }
    }
}

extern "C"
{

    // JNI Function: Initialize NCNN environment
    JNIEXPORT jboolean JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_initNative(JNIEnv *env, jobject /* this */, jobject assetManager)
    {
        if (ncnnInitialized)
        {
            LOGI("NCNN already initialized.");
            return JNI_TRUE;
        }
        LOGI("Initializing NCNN for YOLOv11...");
        auto init_start_time = std::chrono::high_resolution_clock::now();

        // Get AAssetManager
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        if (mgr == nullptr)
        {
            LOGE("Failed to get AAssetManager from Java");
            return JNI_FALSE;
        }

        // Check Vulkan support
        int gpu_count = ncnn::get_gpu_count();
        if (gpu_count > 0)
        {
            useGPU = true;
            LOGI("Vulkan GPU detected. Count: %d. Enabling GPU acceleration.", gpu_count);
            // Initialize Vulkan instance. This is crucial and should happen early.
            ncnn::create_gpu_instance();
        }
        else
        {
            useGPU = false;
            LOGW("No Vulkan GPU detected or NCNN not built with Vulkan support. Using CPU.");
        }

        // Configure NCNN Net options for performance
        yoloNet.opt.lightmode = true;            // Enable light mode
        yoloNet.opt.num_threads = 4;             // Set default CPU threads (less critical if GPU is used)
        yoloNet.opt.use_packing_layout = true;   // Highly recommended for performance on ARM
        yoloNet.opt.use_fp16_packed = useGPU;    // Use FP16 storage on GPU if available
        yoloNet.opt.use_fp16_storage = useGPU;   // Alias for use_fp16_packed
        yoloNet.opt.use_fp16_arithmetic = useGPU;// Use FP16 compute on GPU if available and beneficial
        yoloNet.opt.use_vulkan_compute = useGPU; // Explicitly enable Vulkan compute

        // Optional: Set GPU device index if multiple GPUs exist (usually 0)
        if (useGPU) {
            yoloNet.set_vulkan_device(0);
        }

        ncnnInitialized = true;
        auto init_end_time = std::chrono::high_resolution_clock::now();
        auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end_time - init_start_time);
        LOGI("NCNN initialization complete (took %lld ms). Vulkan enabled: %s", init_duration.count(), useGPU ? "true" : "false");
        return JNI_TRUE;
    }

    // JNI Function: Load the YOLO model
    JNIEXPORT jboolean JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_loadModel(JNIEnv *env, jobject /* this */, jobject assetManager)
    {
        if (!ncnnInitialized)
        {
            LOGE("NCNN not initialized before loading model.");
            return JNI_FALSE;
        }
        if (modelLoaded)
        {
            LOGI("Model already loaded.");
            return JNI_TRUE;
        }
        LOGI("Loading YOLOv11 model...");
        auto load_start_time = std::chrono::high_resolution_clock::now();

        // Get AAssetManager
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        if (mgr == nullptr)
        {
            LOGE("Failed to get AAssetManager from Java for model loading");
            return JNI_FALSE;
        }

        // *** ADJUST THESE FILENAMES TO MATCH YOUR YOLOv11 NCNN MODEL FILES ***
        // Ensure these files exist in your app's `src/main/assets` folder
        const char *param_path = "yolov11.param"; // Example filename
        const char *bin_path = "yolov11.bin";     // Example filename

        int ret_param = -1;
        int ret_bin = -1;
        {
            ncnn::MutexLockGuard guard(yoloNetLock); // Lock before modifying the net
            // Load using the provided asset manager explicitly
            ret_param = yoloNet.load_param(mgr, param_path);
            if (ret_param != 0)
            {
                LOGE("Failed to load model param: %s (Error code: %d)", param_path, ret_param);
                return JNI_FALSE;
            }
            LOGI("Loaded model param: %s", param_path);

            ret_bin = yoloNet.load_model(mgr, bin_path);
            if (ret_bin != 0)
            {
                LOGE("Failed to load model bin: %s (Error code: %d)", bin_path, ret_bin);
                // Clear the net if bin loading fails after param loading succeeded
                yoloNet.clear();
                return JNI_FALSE;
            }
            LOGI("Loaded model bin: %s", bin_path);
        } // Mutex guard released

        modelLoaded = true;
        auto load_end_time = std::chrono::high_resolution_clock::now();
        auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);
        LOGI("YOLOv11 model loading complete (took %lld ms).", load_duration.count());
        return JNI_TRUE;
    }

    // JNI Function: Check if Vulkan (GPU) is being used
    JNIEXPORT jboolean JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_hasVulkan(JNIEnv *env, jobject /* this */)
    {
        // Return the state determined during init
        return (ncnnInitialized && useGPU) ? JNI_TRUE : JNI_FALSE;
    }

    // JNI Function: Perform object detection
    JNIEXPORT jfloatArray JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_detect(JNIEnv *env, jobject /* this */,
                                                            jbyteArray imageBytes, jint imageWidth, jint imageHeight)
    {
        if (!ncnnInitialized || !modelLoaded)
        {
            LOGE("Detection failed: NCNN not initialized or model not loaded.");
            return nullptr; // Return null to indicate failure
        }
        // --- Overall Timing Start ---
        auto total_start_time = std::chrono::high_resolution_clock::now();

        // 1. Get image data from Java byte array
        //    ASSUMPTION: imageBytes contains raw RGBA pixel data.
        jbyte *image_data = env->GetByteArrayElements(imageBytes, nullptr);
        if (image_data == nullptr)
        {
            LOGE("Failed to get image byte array elements.");
            return nullptr;
        }
        const unsigned char *pixel_data = (const unsigned char *)image_data;

        // --- Preprocessing Timing Start ---
        auto preprocess_start_time = std::chrono::high_resolution_clock::now();

        // 2. Preprocessing: Create ncnn::Mat and resize/normalize
        ncnn::Mat input_img;
        // Create Mat from RGBA pixels and resize to model input size (letterboxing/padding handled internally)
        input_img = ncnn::Mat::from_pixels_resize(pixel_data, ncnn::Mat::PIXEL_RGBA, imageWidth, imageHeight,
                                                  YOLOV11_INPUT_WIDTH, YOLOV11_INPUT_HEIGHT);

        // Release the Java byte array *immediately* after creating the ncnn::Mat
        env->ReleaseByteArrayElements(imageBytes, image_data, JNI_ABORT); // Use JNI_ABORT as we copied the data

        if (input_img.empty())
        {
            LOGE("Failed to create ncnn::Mat input_img using from_pixels_resize.");
            return nullptr;
        }

        // Normalize the image (0-255 -> 0-1) - typical for many YOLO models
        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        const float mean_vals[3] = {0.f, 0.f, 0.f}; // No mean subtraction if normalizing to [0, 1]
        input_img.substract_mean_normalize(mean_vals, norm_vals);

        // --- Preprocessing Timing End ---
        auto preprocess_end_time = std::chrono::high_resolution_clock::now();
        auto preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end_time - preprocess_start_time);

        // 3. NCNN Inference
        ncnn::Mat out;
        const char *input_name = nullptr; // Will be determined
        const char *output_name = nullptr; // Will be determined
        std::chrono::microseconds inference_duration(0); // Initialize duration

        {                                            // Scope for extractor and mutex lock
            ncnn::MutexLockGuard guard(yoloNetLock); // Lock for thread safety during inference
            ncnn::Extractor ex = yoloNet.create_extractor();

            // --- Inference Timing Start ---
            auto inference_start_time = std::chrono::high_resolution_clock::now();

            // *** VERIFY INPUT/OUTPUT TENSOR NAMES MATCH YOUR MODEL ***
            input_name = "in0"; // TRY THIS FIRST
            int input_ret = ex.input(input_name, input_img);
            if (input_ret != 0) {
                 input_name = "images"; // Fallback name
                 LOGW("Failed input '%s', trying '%s'", "in0", input_name);
                 input_ret = ex.input(input_name, input_img);
                 if (input_ret != 0) {
                    LOGE("Failed to set input tensor (tried 'in0', 'images'). Error: %d. Check .param file.", input_ret);
                    return nullptr;
                 }
            }

            output_name = "out0"; // TRY THIS FIRST
            int extract_ret = ex.extract(output_name, out);
            if (extract_ret != 0) {
                output_name = "output"; // Fallback name
                LOGW("Failed extract '%s', trying '%s'", "out0", output_name);
                extract_ret = ex.extract(output_name, out);
                 if (extract_ret != 0) {
                    LOGE("Failed to extract output tensor (tried 'out0', 'output'). Error: %d. Check .param file.", extract_ret);
                    return nullptr;
                 }
            }
            // --- Inference Timing End ---
            auto inference_end_time = std::chrono::high_resolution_clock::now();
            inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end_time - inference_start_time);

        } // Mutex guard released, Extractor destroyed

        // Log output tensor dimensions for debugging
        LOGD("Output tensor ('%s') dims: %d, w: %d, h: %d, c: %d, total: %zu, elemsize: %zu",
             output_name, out.dims, out.w, out.h, out.c, out.total(), out.elemsize);

        // --- Postprocessing Timing Start ---
        auto postprocess_start_time = std::chrono::high_resolution_clock::now();

        // 4. Postprocessing
        std::vector<Object> proposals; // Store valid detections before NMS

        // Calculate scaling factors to map detections from model input size back to original image size
        float scale_x = (float)input_img.w / imageWidth;
        float scale_y = (float)input_img.h / imageHeight;
        float scale = std::min(scale_x, scale_y);

        float scaled_w = imageWidth * scale;
        float scaled_h = imageHeight * scale;
        float pad_left = (input_img.w - scaled_w) / 2.0f;
        float pad_top = (input_img.h - scaled_h) / 2.0f;

        int num_detections = 0;
        int num_features = 0;
        const float* output_data = nullptr;

        // Check dimensions assuming [features, num_detections] format after squeezing batch dim
        if (out.dims == 2 && out.w > 4) { // Shape [features, num_detections]
             num_features = out.h;
             num_detections = out.w;
             output_data = (const float*)out.data;
             LOGD("Parsing output format: [%d features, %d detections]", num_features, num_detections);
        }
        // Add check for dims=3 just in case batch dim wasn't squeezed: [1, features, num_detections]
        else if (out.dims == 3 && out.c > 4) {
             num_features = out.h;
             num_detections = out.w;
             output_data = out.channel(0).row(0); // Access data for the first batch
             LOGD("Parsing output format: [1, %d features, %d detections]", num_features, num_detections);
        }
        else {
            LOGE("Unsupported/Unexpected output tensor shape for YOLOv11. Dims=%d, W=%d, H=%d, C=%d", out.dims, out.w, out.h, out.c);
            return nullptr; // Indicate error
        }

        int expected_features = 4 + NUM_CLASSES; // Now 4 + 84 = 88
        if (num_features != expected_features) {
            LOGE("Output tensor feature count (%d) does not match expected features (%d = 4 bbox + %d classes). Check model structure or NUM_CLASSES.",
                 num_features, expected_features, NUM_CLASSES);
            return nullptr; // Indicate error
        }
        if (num_detections <= 0 || output_data == nullptr) {
            LOGW("Output tensor indicates zero potential detections or data pointer is null.");
            // Return empty result array (count=0)
            jfloatArray emptyResult = env->NewFloatArray(1);
            if (emptyResult) {
                float zero = 0.0f;
                env->SetFloatArrayRegion(emptyResult, 0, 1, &zero);
            }
            return emptyResult;
        }

        std::vector<Object> raw_objects;
        raw_objects.reserve(num_detections / 4); // Heuristic reservation

        // Iterate through each potential detection COLUMN (since format is transposed)
        for (int i = 0; i < num_detections; ++i) {
            // Data for detection 'i' is spread across rows, access column-wise

            // Bbox coords are typically the first 4 features (rows 0-3)
            // Class scores are the remaining features (rows 4 to num_features-1)

            // Find the class with the highest score for this detection 'i'
            int best_class_idx = -1;
            float max_class_prob = -1.0f;

            // Access class scores (rows 4 to num_features-1) for the current detection 'i'
            // output_data[row_index * num_detections + column_index]
            for (int c = 0; c < NUM_CLASSES; ++c) {
                float class_prob = output_data[(c + 4) * num_detections + i]; // Access score for class 'c', detection 'i'
                if (class_prob > max_class_prob) {
                    max_class_prob = class_prob;
                    best_class_idx = c;
                }
            }

            // Filter by confidence threshold
            if (max_class_prob >= CONFIDENCE_THRESHOLD) {
                // Extract bounding box coordinates (center_x, center_y, width, height) for detection 'i'
                // Access rows 0-3 for the current detection 'i'
                float cx = output_data[0 * num_detections + i];
                float cy = output_data[1 * num_detections + i];
                float w  = output_data[2 * num_detections + i];
                float h  = output_data[3 * num_detections + i];

                // --- Coordinate transformation and clamping (remains the same) ---
                float x1_net = cx - w / 2.0f;
                float y1_net = cy - h / 2.0f;
                float x2_net = cx + w / 2.0f;
                float y2_net = cy + h / 2.0f;

                float x1_orig = (x1_net - pad_left) / scale;
                float y1_orig = (y1_net - pad_top) / scale;
                float x2_orig = (x2_net - pad_left) / scale;
                float y2_orig = (y2_net - pad_top) / scale;

                x1_orig = std::max(0.0f, std::min((float)imageWidth - 1.0f, x1_orig));
                y1_orig = std::max(0.0f, std::min((float)imageHeight - 1.0f, y1_orig));
                x2_orig = std::max(x1_orig, std::min((float)imageWidth - 1.0f, x2_orig));
                y2_orig = std::max(y1_orig, std::min((float)imageHeight - 1.0f, y2_orig));

                float w_orig = x2_orig - x1_orig;
                float h_orig = y2_orig - y1_orig;

                if (w_orig > 0 && h_orig > 0) {
                    Object obj;
                    obj.x = x1_orig;
                    obj.y = y1_orig;
                    obj.w = w_orig;
                    obj.h = h_orig;
                    obj.label = best_class_idx;
                    obj.prob = max_class_prob;
                    raw_objects.push_back(obj);
                }
                // --- End Coordinate transformation ---
            }
        }
        LOGD("Found %zu raw objects above confidence threshold.", raw_objects.size());

        // Sort by confidence score (descending) before NMS
        std::sort(raw_objects.begin(), raw_objects.end(), [](const Object &a, const Object &b) {
            return a.prob > b.prob;
        });

        // Perform Non-Maximum Suppression
        std::vector<int> picked_indices;
        nms_sorted_bboxes(raw_objects, picked_indices, NMS_THRESHOLD);
        LOGD("NMS resulted in %zu final objects.", picked_indices.size());

        // Collect final proposals based on NMS results
        proposals.reserve(picked_indices.size());
        for (int index : picked_indices) {
            proposals.push_back(raw_objects[index]);
        }

        auto postprocess_end_time = std::chrono::high_resolution_clock::now();
        auto postprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end_time - postprocess_start_time);

        int final_count = proposals.size();
        int result_size = 1 + final_count * 6;
        jfloatArray resultArray = env->NewFloatArray(result_size);
        if (resultArray == nullptr) {
            LOGE("Failed to allocate float array for results (size %d).", result_size);
            return nullptr;
        }

        std::vector<float> resultData(result_size);
        resultData[0] = (float)final_count;

        for (int i = 0; i < final_count; ++i) {
            const Object &obj = proposals[i];
            int offset = 1 + i * 6;
            resultData[offset + 0] = obj.x;
            resultData[offset + 1] = obj.y;
            resultData[offset + 2] = obj.w;
            resultData[offset + 3] = obj.h;
            resultData[offset + 4] = (float)obj.label;
            resultData[offset + 5] = obj.prob;
        }

        env->SetFloatArrayRegion(resultArray, 0, result_size, resultData.data());

        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);

        LOGD("Native detect timing (us): Total=%lld | Preproc=%lld | Infer=%lld | Postproc=%lld | Objects=%d",
             total_duration_us.count(),
             preprocess_duration.count(),
             inference_duration.count(),
             postprocess_duration.count(),
             final_count);

        return resultArray;
    }

    // JNI Function: Release NCNN resources
    JNIEXPORT void JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_releaseNative(JNIEnv *env, jobject /* this */)
    {
        LOGI("Releasing NCNN resources...");
        {
            ncnn::MutexLockGuard guard(yoloNetLock); // Ensure thread safety during cleanup
            yoloNet.clear();                         // Clear the network (releases model data and context)
        }

        if (useGPU)
        {
            ncnn::destroy_gpu_instance(); // Release Vulkan resources
            LOGI("Vulkan GPU instance destroyed.");
        }

        ncnnInitialized = false;
        modelLoaded = false;
        useGPU = false;
        LOGI("NCNN resources released.");
    }

} // extern "C"
