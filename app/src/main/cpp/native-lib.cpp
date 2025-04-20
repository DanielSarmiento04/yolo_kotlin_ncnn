#include <jni.h>
#include <string>
#include <vector>
#include <algorithm> // For std::max, std::min, std::sort
#include <math.h>    // For expf, roundf
#include <numeric>   // For std::iota
#include <chrono>    // For timing

// Android Logging
#include <android/log.h>
#define LOG_TAG "NCNN_Native_YOLOv11"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

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
const float NMS_THRESHOLD = 0.3f;        // IoU threshold for NMS
const float CONFIDENCE_THRESHOLD = 0.4f; // Confidence threshold for filtering detections
const int NUM_CLASSES = 84;              // COCO dataset classes (Adjust if your model differs)

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

// Helper function for Non-Maximum Suppression (NMS)
static inline float intersection_area(const Object &a, const Object &b)
{
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    float width = std::max(0.0f, x2 - x1);
    float height = std::max(0.0f, y2 - y1);
    return width * height;
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
            if (union_area > 1e-6 && (inter_area / union_area) > nms_threshold)
            {
                suppressed[j] = true; // Suppress box j
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
            // Initialize Vulkan instance. This is crucial.
            ncnn::create_gpu_instance();
        }
        else
        {
            useGPU = false;
            LOGI("No Vulkan GPU detected. Using CPU.");
        }

        // Configure NCNN Net options
        yoloNet.opt.lightmode = true;            // Use lightweight mode
        yoloNet.opt.num_threads = 4;             // Use 4 threads as a default
        yoloNet.opt.use_packing_layout = true;   // Recommended for performance
        yoloNet.opt.use_fp16_packed = true;      // Use FP16 storage for reduced memory if supported
        yoloNet.opt.use_fp16_arithmetic = true;  // Use FP16 arithmetic if supported
        yoloNet.opt.use_vulkan_compute = useGPU; // Enable Vulkan if available

        ncnnInitialized = true;
        LOGI("NCNN initialization complete. Vulkan enabled: %s", useGPU ? "true" : "false");
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

        // Get AAssetManager (though NCNN might use the globally set one if opt.use_android_asset_manager is true)
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        if (mgr == nullptr)
        {
            LOGE("Failed to get AAssetManager from Java for model loading");
            return JNI_FALSE;
        }

        // Load model parameters and binary weights from assets
        // *** ADJUST THESE FILENAMES TO MATCH YOUR YOLOv11 NCNN MODEL FILES ***
        // Ensure these files exist in your app's `src/main/assets` folder
        const char *param_path = "yolov11.param"; // Corrected filename
        const char *bin_path = "yolov11.bin";     // Corrected filename

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
        LOGI("YOLOv11 model loading complete.");
        return JNI_TRUE;
    }

    // JNI Function: Check if Vulkan (GPU) is being used
    JNIEXPORT jboolean JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_hasVulkan(JNIEnv *env, jobject /* this */)
    {
        return useGPU ? JNI_TRUE : JNI_FALSE;
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
        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();

        // 1. Get image data from Java byte array
        //    ASSUMPTION: imageBytes contains raw RGBA pixel data.
        jbyte *image_data = env->GetByteArrayElements(imageBytes, nullptr);
        if (image_data == nullptr)
        {
            LOGE("Failed to get image byte array elements.");
            return nullptr;
        }
        const unsigned char *pixel_data = (const unsigned char *)image_data;

        // 2. Preprocessing using from_pixels_resize
        ncnn::Mat input_img;
        // Create Mat from RGBA pixels and resize to model input size
        input_img = ncnn::Mat::from_pixels_resize(pixel_data, ncnn::Mat::PIXEL_RGBA, imageWidth, imageHeight,
                                                  YOLOV11_INPUT_WIDTH, YOLOV11_INPUT_HEIGHT);

        // Release the Java byte array *after* creating the ncnn::Mat
        env->ReleaseByteArrayElements(imageBytes, image_data, JNI_ABORT); // Use JNI_ABORT as we copied the data

        if (input_img.empty())
        {
            LOGE("Failed to create ncnn::Mat input_img using from_pixels_resize.");
            return nullptr;
        }

        // Normalize the image (0-255 -> 0-1)
        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        const float mean_vals[3] = {0.f, 0.f, 0.f}; // No mean subtraction if normalizing to [0, 1]
        input_img.substract_mean_normalize(mean_vals, norm_vals);

        // 3. NCNN Inference
        std::vector<Object> proposals;               // Store valid detections before NMS
        {                                            // Scope for extractor and mutex lock
            ncnn::MutexLockGuard guard(yoloNetLock); // Lock for thread safety
            ncnn::Extractor ex = yoloNet.create_extractor();

            // *** VERIFY THIS INPUT TENSOR NAME MATCHES YOUR MODEL ***
            const char *input_name = "in0"; // TRY THIS NAME FIRST!
            int input_ret = ex.input(input_name, input_img);
            if (input_ret != 0)
            {
                LOGE("Failed to set input tensor '%s'. Error code: %d. Check .param file for correct input name.", input_name, input_ret);
                return nullptr;
            }

            ncnn::Mat out;
            // *** VERIFY THIS OUTPUT TENSOR NAME MATCHES YOUR MODEL ***
            // TRY "out0" FIRST based on logs.
            const char *output_name_attempt1 = "out0";
            int extract_ret = ex.extract(output_name_attempt1, out);

            if (extract_ret != 0)
            {
                // If "out0" fails, try "output" as a fallback.
                const char *output_name_attempt2 = "output";
                LOGW("Failed to extract '%s' (Error code: %d), trying '%s'...", output_name_attempt1, extract_ret, output_name_attempt2);
                extract_ret = ex.extract(output_name_attempt2, out);
                if (extract_ret != 0)
                {
                    // If both attempts fail, log the error and return null.
                    LOGE("Failed to extract output tensor (tried '%s' and '%s'). Last error code: %d. Check .param file for correct output name.", output_name_attempt1, output_name_attempt2, extract_ret);
                    return nullptr;
                }
                // Log success if the second attempt worked
                LOGI("Successfully extracted output tensor using fallback name '%s'.", output_name_attempt2);
                output_name_attempt1 = output_name_attempt2; // Update the name used for logging dimensions
            }

            // Log output tensor dimensions for debugging using the name that succeeded
            LOGI("Output tensor ('%s') dimensions: w=%d, h=%d, c=%d, dims=%d, total=%ld", output_name_attempt1, out.w, out.h, out.c, out.dims, out.total());

            // 4. Postprocessing
            // Calculate scaling factors and padding based on letterboxing
            float scale_w = (float)YOLOV11_INPUT_WIDTH / imageWidth;
            float scale_h = (float)YOLOV11_INPUT_HEIGHT / imageHeight;
            float scale = std::min(scale_w, scale_h); // Scale factor used

            int scaled_w = static_cast<int>(roundf(imageWidth * scale));
            int scaled_h = static_cast<int>(roundf(imageHeight * scale));
            // Calculate padding added during from_pixels_resize (assuming centered)
            int top_pad = (YOLOV11_INPUT_HEIGHT - scaled_h) / 2;
            int left_pad = (YOLOV11_INPUT_WIDTH - scaled_w) / 2;

            // --- Output Parsing Logic ---
            int num_features = out.h;                // Number of features per detection (e.g., 4 bbox + num_classes)
            int num_detections = out.w;              // Number of potential detections (e.g., 8400)
            int expected_features = 4 + NUM_CLASSES; // 4 for bbox (cx, cy, w, h typically) + classes

            if (num_features != expected_features)
            {
                LOGE("Output tensor height (%d) does not match expected features (%d = 4 + %d classes). Check model output layer name and structure.",
                     num_features, expected_features, NUM_CLASSES);
                jfloatArray resultArray = env->NewFloatArray(1); // Array with only count = 0
                if (resultArray)
                {
                    float count = 0.0f;
                    env->SetFloatArrayRegion(resultArray, 0, 1, &count);
                }
                return resultArray;
            }
            if (num_detections <= 0)
            {
                LOGW("Output tensor width (%d) indicates zero potential detections.", num_detections);
                jfloatArray resultArray = env->NewFloatArray(1); // Array with only count = 0
                if (resultArray)
                {
                    float count = 0.0f;
                    env->SetFloatArrayRegion(resultArray, 0, 1, &count);
                }
                return resultArray;
            }

            const float *output_data = out.row(0); // Get pointer to the start of the data

            std::vector<Object> raw_objects;
            raw_objects.reserve(num_detections / 4); // Heuristic reservation

            // Iterate through each potential detection column
            for (int d = 0; d < num_detections; ++d)
            {
                int best_class_idx = -1;
                float max_class_prob = -1.0f;

                const float *class_scores_ptr = output_data + 4 * num_detections;
                for (int c = 0; c < NUM_CLASSES; ++c)
                {
                    float class_prob = class_scores_ptr[c * num_detections + d];
                    if (class_prob > max_class_prob)
                    {
                        max_class_prob = class_prob;
                        best_class_idx = c;
                    }
                }

                if (max_class_prob >= CONFIDENCE_THRESHOLD)
                {
                    float cx = output_data[0 * num_detections + d];
                    float cy = output_data[1 * num_detections + d];
                    float w = output_data[2 * num_detections + d];
                    float h = output_data[3 * num_detections + d];

                    float x1_padded = cx - w / 2.0f;
                    float y1_padded = cy - h / 2.0f;

                    float x1_orig = (x1_padded - left_pad) / scale;
                    float y1_orig = (y1_padded - top_pad) / scale;
                    float w_orig = w / scale;
                    float h_orig = h / scale;

                    x1_orig = std::max(0.0f, std::min((float)imageWidth - 1.0f, x1_orig));
                    y1_orig = std::max(0.0f, std::min((float)imageHeight - 1.0f, y1_orig));
                    w_orig = std::min((float)imageWidth - x1_orig, w_orig);
                    h_orig = std::min((float)imageHeight - y1_orig, h_orig);

                    Object obj;
                    obj.x = x1_orig;
                    obj.y = y1_orig;
                    obj.w = w_orig;
                    obj.h = h_orig;
                    obj.label = best_class_idx;
                    obj.prob = max_class_prob;

                    if (obj.w > 0 && obj.h > 0)
                    {
                        raw_objects.push_back(obj);
                    }
                }
            }
            LOGI("Found %zu raw objects above confidence threshold.", raw_objects.size());

            std::sort(raw_objects.begin(), raw_objects.end(), [](const Object &a, const Object &b)
                      { return a.prob > b.prob; });

            std::vector<int> picked_indices;
            nms_sorted_bboxes(raw_objects, picked_indices, NMS_THRESHOLD);
            LOGI("NMS resulted in %zu final objects.", picked_indices.size());

            proposals.reserve(picked_indices.size());
            for (int index : picked_indices)
            {
                proposals.push_back(raw_objects[index]);
            }
        }

        int final_count = proposals.size();
        LOGI("Native detection found %d objects after NMS.", final_count);
        int result_size = 1 + final_count * 6;
        jfloatArray resultArray = env->NewFloatArray(result_size);
        if (resultArray == nullptr)
        {
            LOGE("Failed to allocate float array for results.");
            return nullptr;
        }

        std::vector<float> resultData(result_size);
        resultData[0] = (float)final_count;

        for (int i = 0; i < final_count; ++i)
        {
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

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        LOGI("Native detection total time: %lld ms", duration.count());

        return resultArray;
    }

    // JNI Function: Release NCNN resources
    JNIEXPORT void JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_releaseNative(JNIEnv *env, jobject /* this */)
    {
        LOGI("Releasing NCNN resources...");
        {
            ncnn::MutexLockGuard guard(yoloNetLock); // Ensure thread safety during cleanup
            yoloNet.clear();                         // Clear the network (releases model data)
        }

        if (useGPU)
        {
            ncnn::destroy_gpu_instance();
            LOGI("Vulkan GPU instance destroyed.");
        }

        ncnnInitialized = false;
        modelLoaded = false;
        useGPU = false;
        LOGI("NCNN resources released.");
    }

} // extern "C"
