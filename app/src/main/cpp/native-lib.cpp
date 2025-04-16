#include <jni.h>
#include <string>
#include <vector>
#include <algorithm> // For std::max, std::min
#include <math.h>    // For expf

// Android Logging
#include <android/log.h>
#define LOG_TAG "NCNN_Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Asset Manager (for loading models from assets)
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

// NCNN Headers
#include "ncnn/net.h"
#include "ncnn/gpu.h"
#include "ncnn/layer.h" // Required for custom layers if any

// Global NCNN Net instance and state variables
static ncnn::Net yoloNet;
static ncnn::Mutex yoloNetLock; // For thread safety if calling detect from multiple threads
static bool ncnnInitialized = false;
static bool modelLoaded = false;
static bool useGPU = false; // Whether Vulkan will be used

// YOLOv5 constants (adjust if using a different model)
const int YOLOV5_INPUT_WIDTH = 640;
const int YOLOV5_INPUT_HEIGHT = 640;
const float NMS_THRESHOLD = 0.45f;
const float CONFIDENCE_THRESHOLD = 0.25f;
const int NUM_CLASSES = 80; // COCO dataset

// Structure to hold detection results
struct Object {
    float x;      // Top-left corner x (relative to original image width)
    float y;      // Top-left corner y (relative to original image height)
    float w;      // Width (relative to original image width)
    float h;      // Height (relative to original image height)
    int label;    // Class index
    float prob;   // Confidence score
};

// Helper function for Non-Maximum Suppression (NMS)
static inline float intersection_area(const Object& a, const Object& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    float width = std::max(0.0f, x2 - x1);
    float height = std::max(0.0f, y2 - y1);
    return width * height;
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold) {
    picked.clear();
    const int n = objects.size();
    if (n == 0) return;

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = objects[i].w * objects[i].h;
    }

    // Sort by probability (descending) - assumes input `objects` are already sorted if needed
    // If not pre-sorted, you'd need to sort here based on `prob`.

    for (int i = 0; i < n; i++) {
        const Object& a = objects[i];
        bool keep = true;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object& b = objects[picked[j]];

            // Intersection over Union (IoU)
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float iou = inter_area / union_area; // Avoid division by zero if union_area is 0
            if (union_area > 1e-6 && (inter_area / union_area) > nms_threshold) {
                keep = false;
                break; // Stop checking intersection with others for object 'a'
            }
        }
        if (keep) {
            picked.push_back(i);
        }
    }
}

// Custom YoloV5Focus layer implementation (if needed, check your .param file)
// If your yolov5s.param does NOT contain a "YoloV5Focus" layer type, you can remove this.
// Many newer YOLOv5 exports replace Focus with a standard Conv layer.
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus() { one_blob_only = true; }
    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;
        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty()) return -100;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++) {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);
            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    *outptr = *ptr;
                    outptr += 1;
                    ptr += 2;
                }
                ptr += w; // Move to the next row in the input blob (stride is w)
            }
        }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(YoloV5Focus) // Macro to register the layer creator

extern "C" {

// JNI Function: Initialize NCNN environment
JNIEXPORT jboolean JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_initNative(JNIEnv *env, jobject /* this */, jobject assetManager) {
    if (ncnnInitialized) {
        LOGI("NCNN already initialized.");
        return JNI_TRUE;
    }
    LOGI("Initializing NCNN...");

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
        LOGI("Vulkan GPU detected. Count: %d", gpu_count);
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

    // Register custom layers IF NEEDED (check your .param file)
    // If your model uses custom layers like YoloV5Focus, register them.
    // If not, you can comment this out.
    int ret = yoloNet.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    if (ret != 0) {
//        LOGW("Failed to register custom layer YoloV5Focus (maybe not needed?). Error code: %d", ret);
        // Continue even if registration fails, maybe the model doesn't use it.
    } else {
        LOGI("Custom layer YoloV5Focus registered.");
    }


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
    LOGI("Loading YOLO model...");

    // Get AAssetManager
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    if (mgr == nullptr) {
        LOGE("Failed to get AAssetManager from Java for model loading");
        return JNI_FALSE;
    }

    // Load model parameters and binary weights from assets
    // Ensure these filenames match exactly what's in your app/src/main/assets folder
    const char *param_path = "yolov5s.param";
    const char *bin_path = "yolov5s.bin";

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
    LOGI("YOLO model loading complete.");
    return JNI_TRUE;
}

// JNI Function: Check if Vulkan (GPU) is being used
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

    // 1. Get image data from Java byte array (assuming RGBA format)
    jbyte *image_data = env->GetByteArrayElements(imageBytes, nullptr);
    if (image_data == nullptr) {
        LOGE("Failed to get image byte array elements.");
        return nullptr;
    }
    // Wrap data in ncnn::Mat. The 4 is for RGBA channels.
    // We create it directly from the RGBA data.
    ncnn::Mat img_rgba = ncnn::Mat(imageWidth, imageHeight, (void*)image_data, (size_t)4, 4);

    // 2. Preprocessing
    // Convert RGBA to RGB (NCNN typically expects BGR or RGB)
    ncnn::Mat img_rgb;
//    ncnn::cvtColor(img_rgba, img_rgb, ncnn::COLOR_RGBA2RGB);

    // Calculate scaling factor and padding for letterboxing
    float scale_w = (float)YOLOV5_INPUT_WIDTH / imageWidth;
    float scale_h = (float)YOLOV5_INPUT_HEIGHT / imageHeight;
    float scale = std::min(scale_w, scale_h); // Use the smaller scale factor to fit inside

    int scaled_w = static_cast<int>(imageWidth * scale);
    int scaled_h = static_cast<int>(imageHeight * scale);

    // Resize the image
    ncnn::Mat resized_img;
    ncnn::resize_bilinear(img_rgb, resized_img, scaled_w, scaled_h);

    // Create padded image (letterbox)
    ncnn::Mat input_img;
    // Calculate padding offsets
    int top_pad = (YOLOV5_INPUT_HEIGHT - scaled_h) / 2;
    int bottom_pad = YOLOV5_INPUT_HEIGHT - scaled_h - top_pad;
    int left_pad = (YOLOV5_INPUT_WIDTH - scaled_w) / 2;
    int right_pad = YOLOV5_INPUT_WIDTH - scaled_w - left_pad;

    // Apply padding (value 114 is common for YOLO, representing gray)
    ncnn::copy_make_border(resized_img, input_img, top_pad, bottom_pad, left_pad, right_pad, ncnn::BORDER_CONSTANT, 114.f);

    // Normalize the image (0-255 -> 0-1)
    // NCNN handles mean/norm internally if specified in the .param file,
    // but manual normalization is often done for YOLO models.
    // Check your model's requirements. This assumes normalization to [0, 1].
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    const float mean_vals[3] = {0.f, 0.f, 0.f}; // No mean subtraction if normalizing to [0, 1]
    input_img.substract_mean_normalize(mean_vals, norm_vals);

    // Release the Java byte array *after* converting to ncnn::Mat
    env->ReleaseByteArrayElements(imageBytes, image_data, JNI_ABORT); // Use JNI_ABORT if no changes were made

    // 3. NCNN Inference
    std::vector<Object> proposals; // Store raw detections before NMS
    { // Scope for extractor and mutex lock
        ncnn::MutexLockGuard guard(yoloNetLock); // Lock for thread safety
        ncnn::Extractor ex = yoloNet.create_extractor();
        // Set extractor options if needed (e.g., ex.set_num_threads(2)); inherits from net by default
        ex.set_vulkan_compute(useGPU);

        // Input tensor name "images" is common for YOLOv5 ONNX exports
        ex.input("images", input_img);

        // Output tensor name "output" is also common
        ncnn::Mat out;
        ex.extract("output", out); // Or the actual output node name from your .param file

        // 4. Postprocessing
        // out shape is typically [1, num_predictions, 5 + num_classes]
        // num_predictions = 25200 for 640x640 input (80x80 + 40x40 + 20x20 anchors * 3 scales)
        // Format: [cx, cy, w, h, box_confidence, class1_prob, class2_prob, ...]
        int num_predictions = out.h; // Should be 25200
        int num_outputs = out.w;     // Should be 85 (5 + 80 classes)

        if (num_outputs != 5 + NUM_CLASSES) {
             LOGE("Output tensor width (%d) does not match expected size (5 + %d classes)", num_outputs, NUM_CLASSES);
             return nullptr;
        }

        for (int i = 0; i < num_predictions; i++) {
            const float* row = out.row(i);
            float box_confidence = row[4];

            if (box_confidence >= CONFIDENCE_THRESHOLD) {
                // Find the class with the highest score
                int best_class_idx = 0;
                float max_class_prob = 0.0f;
                for (int j = 0; j < NUM_CLASSES; j++) {
                    float class_prob = row[5 + j];
                    if (class_prob > max_class_prob) {
                        max_class_prob = class_prob;
                        best_class_idx = j;
                    }
                }

                float final_confidence = box_confidence * max_class_prob;

                if (final_confidence >= CONFIDENCE_THRESHOLD) {
                    // Decode bounding box coordinates (center_x, center_y, width, height)
                    // These are relative to the 640x640 input image
                    float cx = row[0];
                    float cy = row[1];
                    float w = row[2];
                    float h = row[3];

                    // Convert from center format to top-left format and scale back to original image dimensions
                    float x1 = (cx - w / 2.0f - left_pad) / scale;
                    float y1 = (cy - h / 2.0f - top_pad) / scale;
                    float x2 = (cx + w / 2.0f - left_pad) / scale;
                    float y2 = (cy + h / 2.0f - top_pad) / scale;

                    // Clamp coordinates to image bounds
                    x1 = std::max(0.0f, std::min((float)imageWidth, x1));
                    y1 = std::max(0.0f, std::min((float)imageHeight, y1));
                    x2 = std::max(0.0f, std::min((float)imageWidth, x2));
                    y2 = std::max(0.0f, std::min((float)imageHeight, y2));

                    Object obj;
                    obj.x = x1; // Top-left x
                    obj.y = y1; // Top-left y
                    obj.w = x2 - x1; // Width
                    obj.h = y2 - y1; // Height
                    obj.label = best_class_idx;
                    obj.prob = final_confidence;
                    proposals.push_back(obj);
                }
            }
        }
    } // End scope for extractor and mutex lock

    // Apply Non-Maximum Suppression (NMS)
    std::vector<int> picked_indices;
    // Sort proposals by confidence before NMS (important!)
    std::sort(proposals.begin(), proposals.end(), [](const Object& a, const Object& b) {
        return a.prob > b.prob;
    });
    nms_sorted_bboxes(proposals, picked_indices, NMS_THRESHOLD);

    // 5. Format results for Java
    // Format: [count, x1, y1, w1, h1, label1, conf1, x2, y2, w2, h2, label2, conf2, ...]
    int final_count = picked_indices.size();
    int result_size = 1 + final_count * 6; // 1 for count, 6 floats per detection
    jfloatArray resultArray = env->NewFloatArray(result_size);
    if (resultArray == nullptr) {
        LOGE("Failed to allocate float array for results.");
        return nullptr;
    }

    std::vector<float> resultData(result_size);
    resultData[0] = (float)final_count;

    for (int i = 0; i < final_count; ++i) {
        const Object& obj = proposals[picked_indices[i]];
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


// JNI Function: Release NCNN resources
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
