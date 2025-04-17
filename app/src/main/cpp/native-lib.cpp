#include <jni.h>
#include <string>
#include <vector>
#include <algorithm> // For std::max, std::min, std::sort
#include <math.h>    // For expf, roundf
#include <numeric>   // For std::iota

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
// Use thresholds from YOLO11.hpp reference
const float NMS_THRESHOLD = 0.3f;         // IoU threshold for NMS
const float CONFIDENCE_THRESHOLD = 0.4f;  // Confidence threshold for filtering detections
const int NUM_CLASSES = 80; // COCO dataset classes (Adjust if your model differs)

// Structure to hold detection results
struct Object {
    float x;      // Top-left corner x (relative to original image width)
    float y;      // Top-left corner y (relative to original image height)
    float w;      // Width (relative to original image width)
    float h;      // Height (relative to original image height)
    int label;    // Class index
    float prob;   // Confidence score
};

// Helper function for Non-Maximum Suppression (NMS) - unchanged
static inline float intersection_area(const Object& a, const Object& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    float width = std::max(0.0f, x2 - x1);
    float height = std::max(0.0f, y2 - y1);
    return width * height;
}

// NMS function - assumes input `objects` are pre-sorted by confidence
static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    const int n = objects.size();
    if (n == 0) return;

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = objects[i].w * objects[i].h;
    }

    std::vector<bool> suppressed(n, false); // Keep track of suppressed boxes

    for (int i = 0; i < n; i++) {
        if (suppressed[i]) {
            continue; // Skip if already suppressed
        }
        picked.push_back(i); // Pick the current box
        const Object& a = objects[i];

        for (int j = i + 1; j < n; j++) {
            if (suppressed[j]) {
                continue; // Skip if already suppressed
            }
            const Object& b = objects[j];

            // Intersection over Union (IoU)
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[j] - inter_area;

            if (union_area > 1e-6 && (inter_area / union_area) > nms_threshold) {
                suppressed[j] = true; // Suppress box j
            }
        }
    }
}

extern "C" {

// JNI Function: Initialize NCNN environment
JNIEXPORT jboolean JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_initNative(JNIEnv *env, jobject /* this */, jobject assetManager) {
    if (ncnnInitialized) {
        LOGI("NCNN already initialized.");
        return JNI_TRUE;
    }
    LOGI("Initializing NCNN for YOLOv11...");

    // Get AAssetManager
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    if (mgr == nullptr) {
        LOGE("Failed to get AAssetManager from Java");
        return JNI_FALSE;
    }

    // Check Vulkan support
    int gpu_count = ncnn::get_gpu_count();
    if (gpu_count > 0) {
        useGPU = true;
        LOGI("Vulkan GPU detected. Count: %d. Enabling GPU acceleration.", gpu_count);
        // Initialize Vulkan instance. This is crucial.
        ncnn::create_gpu_instance();
    } else {
        useGPU = false;
        LOGI("No Vulkan GPU detected. Using CPU.");
    }

    // Configure NCNN Net options
    yoloNet.opt.lightmode = true; // Use lightweight mode
    yoloNet.opt.num_threads = 4;  // Adjust based on device core count
    yoloNet.opt.use_packing_layout = true; // Recommended for performance
    yoloNet.opt.use_fp16_packed = true;    // Use FP16 storage for reduced memory
    yoloNet.opt.use_fp16_arithmetic = true;// Use FP16 arithmetic if supported (check device capabilities)
    yoloNet.opt.use_vulkan_compute = useGPU; // Enable Vulkan if available

    // No custom layer registration needed for standard YOLOv11

    ncnnInitialized = true;
    LOGI("NCNN initialization complete. Vulkan enabled: %s", useGPU ? "true" : "false");
    return JNI_TRUE;
}

// JNI Function: Load the YOLO model
JNIEXPORT jboolean JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_loadModel(JNIEnv *env, jobject /* this */, jobject assetManager) {
    if (!ncnnInitialized) {
        LOGE("NCNN not initialized before loading model.");
        return JNI_FALSE;
    }
    if (modelLoaded) {
        LOGI("Model already loaded.");
        return JNI_TRUE;
    }
    LOGI("Loading YOLOv11 model...");

    // Get AAssetManager
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    if (mgr == nullptr) {
        LOGE("Failed to get AAssetManager from Java for model loading");
        return JNI_FALSE;
    }

    // Load model parameters and binary weights from assets
    // *** ADJUST THESE FILENAMES TO MATCH YOUR YOLOv11 NCNN MODEL FILES ***
    const char *param_path = "yolov11.param"; // Example filename
    const char *bin_path = "yolov11.bin";   // Example filename

    int ret_param = yoloNet.load_param(mgr, param_path);
    if (ret_param != 0) {
        LOGE("Failed to load model param: %s (Error code: %d)", param_path, ret_param);
        return JNI_FALSE;
    }
    LOGI("Loaded model param: %s", param_path);

    int ret_bin = yoloNet.load_model(mgr, bin_path);
    if (ret_bin != 0) {
        LOGE("Failed to load model bin: %s (Error code: %d)", bin_path, ret_bin);
        // Clear the net if bin loading fails after param loading succeeded
        yoloNet.clear();
        return JNI_FALSE;
    }
    LOGI("Loaded model bin: %s", bin_path);

    modelLoaded = true;
    LOGI("YOLOv11 model loading complete.");
    return JNI_TRUE;
}

// JNI Function: Check if Vulkan (GPU) is being used - unchanged
JNIEXPORT jboolean JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_hasVulkan(JNIEnv *env, jobject /* this */) {
    return useGPU ? JNI_TRUE : JNI_FALSE;
}

// JNI Function: Perform object detection
JNIEXPORT jfloatArray JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_detect(JNIEnv *env, jobject /* this */,
                                                        jbyteArray imageBytes, jint imageWidth, jint imageHeight) {
    if (!ncnnInitialized || !modelLoaded) {
        LOGE("Detection failed: NCNN not initialized or model not loaded.");
        return nullptr; // Return null to indicate failure
    }

    // 1. Get image data from Java byte array
    //    ASSUMPTION: imageBytes contains raw RGBA pixel data.
    //    If it's encoded (JPEG/PNG), you need to decode it first.
    //    ncnn::Mat::from_pixels can handle various formats if needed.
    jbyte *image_data = env->GetByteArrayElements(imageBytes, nullptr);
    if (image_data == nullptr) {
        LOGE("Failed to get image byte array elements.");
        return nullptr;
    }

    // Create ncnn::Mat from RGBA data. Note: NCNN uses bytes_per_pixel * width for stride.
    // Use ncnn::Mat::from_pixels for better format handling.
    ncnn::Mat img_rgba = ncnn::Mat::from_pixels((const unsigned char*)image_data, ncnn::Mat::PIXEL_RGBA, imageWidth, imageHeight);
    if (img_rgba.empty()) {
        LOGE("Failed to create ncnn::Mat from RGBA pixels.");
        env->ReleaseByteArrayElements(imageBytes, image_data, JNI_ABORT);
        return nullptr;
    }

    // Release the Java byte array *after* creating the ncnn::Mat
    env->ReleaseByteArrayElements(imageBytes, image_data, JNI_ABORT); // Use JNI_ABORT as we copied the data

    // 2. Preprocessing
    // Convert RGBA to RGB (YOLO models typically expect RGB)
    ncnn::Mat img_rgb;
    // Use ncnn::convert_color and ncnn::Mat::PIXEL_RGBA2RGB
    ncnn::convert_color(img_rgba, img_rgb, ncnn::Mat::PIXEL_RGBA2RGB); // Or PIXEL_RGBA2BGR if model expects BGR

    // Calculate scaling factor and padding for letterboxing
    float scale_w = (float)YOLOV11_INPUT_WIDTH / img_rgb.w;
    float scale_h = (float)YOLOV11_INPUT_HEIGHT / img_rgb.h;
    float scale = std::min(scale_w, scale_h); // Use the smaller scale factor to fit inside

    int scaled_w = static_cast<int>(roundf(img_rgb.w * scale));
    int scaled_h = static_cast<int>(roundf(img_rgb.h * scale));

    // Resize the image
    ncnn::Mat resized_img;
    ncnn::resize_bilinear(img_rgb, resized_img, scaled_w, scaled_h);

    // Create padded image (letterbox)
    ncnn::Mat input_img;
    // Calculate padding offsets
    int top_pad = (YOLOV11_INPUT_HEIGHT - scaled_h) / 2;
    int bottom_pad = YOLOV11_INPUT_HEIGHT - scaled_h - top_pad;
    int left_pad = (YOLOV11_INPUT_WIDTH - scaled_w) / 2;
    int right_pad = YOLOV11_INPUT_WIDTH - scaled_w - left_pad;

    // Apply padding (value 114 is common for YOLO, representing gray)
    // Ensure the padding value matches the model's training if different.
    ncnn::copy_make_border(resized_img, input_img, top_pad, bottom_pad, left_pad, right_pad, ncnn::BORDER_CONSTANT, 114.f);

    // Normalize the image (0-255 -> 0-1)
    // NCNN handles mean/norm internally if specified in the .param file (sub/norm ops),
    // but manual normalization is often done for YOLO models.
    // This assumes normalization to [0, 1]. Adjust if your model requires different normalization.
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    const float mean_vals[3] = {0.f, 0.f, 0.f}; // No mean subtraction if normalizing to [0, 1]
    input_img.substract_mean_normalize(mean_vals, norm_vals);


    // 3. NCNN Inference
    std::vector<Object> proposals; // Store valid detections before NMS
    { // Scope for extractor and mutex lock
        ncnn::MutexLockGuard guard(yoloNetLock); // Lock for thread safety
        ncnn::Extractor ex = yoloNet.create_extractor();
        ex.set_vulkan_compute(useGPU);

        // Input tensor name "images" is common for YOLO ONNX exports converted to NCNN
        // Verify this name matches your .param file's input layer name
        ex.input("images", input_img); // Adjust "images" if needed

        // Output tensor name "output" is also common
        // Verify this name matches your .param file's output layer name
        ncnn::Mat out;
        ex.extract("output", out); // Adjust "output" if needed

        // 4. Postprocessing - Adapted for YOLOv11 output format [features, detections]
        //    Expected 'out' shape: [w=num_detections, h=num_features, c=1]
        //    num_features = 4 (bbox) + num_classes
        //    Example: For 640x640 input, num_detections might be 8400.
        //             num_features = 4 + 80 = 84.
        //             out shape: [w=8400, h=84, c=1]

        int num_features = out.h; // e.g., 84
        int num_detections = out.w; // e.g., 8400
        int expected_features = 4 + NUM_CLASSES; // 4 for bbox (cx, cy, w, h)

        if (num_features != expected_features) {
             LOGE("Output tensor height (%d) does not match expected features (%d = 4 + %d classes)",
                  num_features, expected_features, NUM_CLASSES);
             return nullptr;
        }
        if (num_detections <= 0) {
            LOGW("Output tensor width (%d) indicates zero detections.", num_detections);
            // Return empty array instead of null
            jfloatArray resultArray = env->NewFloatArray(1); // Array with only count = 0
            if (resultArray) {
                float count = 0.0f;
                env->SetFloatArrayRegion(resultArray, 0, 1, &count);
            }
            return resultArray;
        }

        // Access the raw output data
        // NCNN Mat stores data channel by channel, then row by row within a channel.
        // For a 2D-like Mat (c=1), out.channel(0) gives the pointer to the start.
        // Data layout is [feature0_det0, feature0_det1, ..., feature1_det0, ...]
        const float* output_data = out.channel(0);

        std::vector<Object> raw_objects;
        raw_objects.reserve(num_detections); // Reserve space

        for (int d = 0; d < num_detections; ++d) {
            // Find the class with the highest score for this detection 'd'
            int best_class_idx = -1;
            float max_class_prob = -1.0f; // Use -1 to ensure any valid score is higher

            // Class scores start after the 4 bounding box values
            // Access pattern: output_data[feature_index * num_detections + detection_index]
            const float* class_scores_ptr = output_data + 4 * num_detections;

            for (int c = 0; c < NUM_CLASSES; ++c) {
                float class_prob = class_scores_ptr[c * num_detections + d];
                if (class_prob > max_class_prob) {
                    max_class_prob = class_prob;
                    best_class_idx = c;
                }
            }

            // Check if the highest class score meets the confidence threshold
            if (max_class_prob >= CONFIDENCE_THRESHOLD) {
                // Extract bounding box coordinates (center_x, center_y, width, height)
                // These are relative to the padded 640x640 input image
                float cx = output_data[0 * num_detections + d];
                float cy = output_data[1 * num_detections + d];
                float w = output_data[2 * num_detections + d];
                float h = output_data[3 * num_detections + d];

                // Convert from center format to top-left format (relative to 640x640)
                float x1_padded = cx - w / 2.0f;
                float y1_padded = cy - h / 2.0f;
                float x2_padded = cx + w / 2.0f;
                float y2_padded = cy + h / 2.0f;

                // Scale back to original image dimensions, removing padding
                // Invert the letterboxing process
                float x1_orig = (x1_padded - left_pad) / scale;
                float y1_orig = (y1_padded - top_pad) / scale;
                float x2_orig = (x2_padded - left_pad) / scale;
                float y2_orig = (y2_padded - top_pad) / scale;

                // Clamp coordinates to original image bounds
                x1_orig = std::max(0.0f, std::min((float)imageWidth - 1.0f, x1_orig));
                y1_orig = std::max(0.0f, std::min((float)imageHeight - 1.0f, y1_orig));
                x2_orig = std::max(0.0f, std::min((float)imageWidth - 1.0f, x2_orig));
                y2_orig = std::max(0.0f, std::min((float)imageHeight - 1.0f, y2_orig));

                Object obj;
                obj.x = x1_orig; // Top-left x
                obj.y = y1_orig; // Top-left y
                obj.w = x2_orig - x1_orig; // Width
                obj.h = y2_orig - y1_orig; // Height
                obj.label = best_class_idx;
                obj.prob = max_class_prob; // Use the max class score as confidence

                // Ensure width and height are non-negative
                if (obj.w >= 0 && obj.h >= 0) {
                    raw_objects.push_back(obj);
                }
            }
        }
        // Sort proposals by confidence (descending) before NMS
        std::sort(raw_objects.begin(), raw_objects.end(), [](const Object& a, const Object& b) {
            return a.prob > b.prob;
        });

        // Apply Non-Maximum Suppression (NMS)
        std::vector<int> picked_indices;
        nms_sorted_bboxes(raw_objects, picked_indices, NMS_THRESHOLD);

        // Prepare final results
        proposals.reserve(picked_indices.size());
        for (int index : picked_indices) {
            proposals.push_back(raw_objects[index]);
        }

    } // End scope for extractor and mutex lock


    // 5. Format results for Java
    // Format: [count, x1, y1, w1, h1, label1, conf1, x2, y2, w2, h2, label2, conf2, ...]
    int final_count = proposals.size();
    int result_size = 1 + final_count * 6; // 1 for count, 6 floats per detection
    jfloatArray resultArray = env->NewFloatArray(result_size);
    if (resultArray == nullptr) {
        LOGE("Failed to allocate float array for results.");
        return nullptr;
    }

    std::vector<float> resultData(result_size);
    resultData[0] = (float)final_count;

    for (int i = 0; i < final_count; ++i) {
        const Object& obj = proposals[i];
        int offset = 1 + i * 6;
        resultData[offset + 0] = obj.x;
        resultData[offset + 1] = obj.y;
        resultData[offset + 2] = obj.w;
        resultData[offset + 3] = obj.h;
        resultData[offset + 4] = (float)obj.label;
        resultData[offset + 5] = obj.prob;
    }

    // Copy data to the Java float array
    env->SetFloatArrayRegion(resultArray, 0, result_size, resultData.data());

    return resultArray;
}


// JNI Function: Release NCNN resources - unchanged
JNIEXPORT void JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_releaseNative(JNIEnv *env, jobject /* this */) {
    LOGI("Releasing NCNN resources...");
    {
        ncnn::MutexLockGuard guard(yoloNetLock); // Ensure thread safety during cleanup
        yoloNet.clear(); // Clear the network (releases model data)
    }

    // Destroy the Vulkan instance if it was created
    if (useGPU) {
        ncnn::destroy_gpu_instance();
        LOGI("Vulkan GPU instance destroyed.");
    }

    ncnnInitialized = false;
    modelLoaded = false;
    useGPU = false;
    LOGI("NCNN resources released.");
}

} // extern "C"
