#include <jni.h>
#include <android/log.h>
#include <android/asset_manager_jni.h>
#include "ncnn/net.h"
#include "ncnn/gpu.h"
#include <vector>
#include <algorithm>

#define LOG_TAG "NCNN"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Create an instance of the NCNN Net that you'll use for inference.
static ncnn::Net yoloNet;
static ncnn::Mutex lock;
static bool hasGPU = false;
static bool modelLoaded = false;

class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;
                    outptr += 1;
                    ptr += 2;
                }
                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)

struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    float x1 = std::max(a.x - a.w/2, b.x - b.w/2);
    float y1 = std::max(a.y - a.h/2, b.y - b.h/2);
    float x2 = std::min(a.x + a.w/2, b.x + b.w/2);
    float y2 = std::min(a.y + a.h/2, b.y + b.h/2);
    
    float area = (x2 - x1) * (y2 - y1);
    return area > 0 ? area : 0;
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].w * objects[i].h;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_initNative(JNIEnv* env, jobject /* this */, jobject assetManager) {
    LOGI("Native init called");
    
    // Convert Java AssetManager to native AAssetManager
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) {
        LOGE("Failed to get AAssetManager");
        return JNI_FALSE;
    }
    
    // Initialize Vulkan if supported
    ncnn::create_gpu_instance();
    hasGPU = ncnn::get_gpu_count() > 0;
    
    LOGI("Vulkan support: %s", hasGPU ? "available" : "unavailable");
    
    // Configure NCNN options
    yoloNet.opt.num_threads = 4;  // CPU threads for non-Vulkan operations
    yoloNet.opt.lightmode = true; // Enable light mode for faster inference
    
    // Use Vulkan if available
    if (hasGPU) {
        yoloNet.opt.use_vulkan_compute = true;
        LOGI("Vulkan enabled");
    } else {
        yoloNet.opt.use_vulkan_compute = false;
        LOGI("Vulkan disabled, using CPU");
    }
    
    // Register custom layers
    yoloNet.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_loadModel(JNIEnv* env, jobject /* this */, jobject assetManager) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) {
        LOGE("Failed to get AAssetManager");
        return JNI_FALSE;
    }

    // Loading the YOLO model
    yoloNet.clear();
    int ret1 = yoloNet.load_param(mgr, "yolov5s.param");
    int ret2 = yoloNet.load_model(mgr, "yolov5s.bin");
    
    modelLoaded = (ret1 == 0 && ret2 == 0);
    
    if (modelLoaded) {
        LOGI("YOLOv5 model loaded successfully");
    } else {
        LOGE("Failed to load YOLOv5 model: param=%d, bin=%d", ret1, ret2);
    }
    
    return modelLoaded ? JNI_TRUE : JNI_FALSE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_hasVulkan(JNIEnv* env, jobject /* this */) {
    return hasGPU ? JNI_TRUE : JNI_FALSE;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_detect(JNIEnv* env, jobject /* this */, 
                                                        jbyteArray imageBytes, jint width, jint height) {
    if (!modelLoaded) {
        LOGE("Model not loaded");
        return nullptr;
    }
    
    // Lock to ensure thread safety
    ncnn::MutexLockGuard g(lock);
    
    // Get image data from Java
    jbyte* data = env->GetByteArrayElements(imageBytes, nullptr);
    if (!data) {
        LOGE("Failed to get image data");
        return nullptr;
    }
    
    // Convert RGBA to RGB
    ncnn::Mat in = ncnn::Mat(width, height, 3);
    const unsigned char* rgba = reinterpret_cast<const unsigned char*>(data);
    
    #pragma omp parallel for num_threads(4)
    for (int y = 0; y < height; y++) {
        const unsigned char* rgba_row = rgba + y * width * 4;
        unsigned char* rgb_row = in.row<unsigned char>(y);
        
        for (int x = 0; x < width; x++) {
            rgb_row[0] = rgba_row[0]; // R
            rgb_row[1] = rgba_row[1]; // G
            rgb_row[2] = rgba_row[2]; // B
            
            rgba_row += 4;
            rgb_row += 3;
        }
    }
    
    // Release byte array
    env->ReleaseByteArrayElements(imageBytes, data, JNI_ABORT);
    
    // YOLOv5 prefers 640x640 input
    const int target_size = 640;
    
    int img_w = width;
    int img_h = height;
    
    // Scale to maintain aspect ratio
    float scale = 1.0f;
    if (img_w > img_h) {
        scale = (float)target_size / img_w;
    } else {
        scale = (float)target_size / img_h;
    }
    
    int scaled_w = int(img_w * scale);
    int scaled_h = int(img_h * scale);
    
    // Resize
    ncnn::Mat in_resized;
    ncnn::resize_bilinear(in, in_resized, scaled_w, scaled_h);
    
    // Pad to square
    ncnn::Mat in_pad(target_size, target_size, 3);
    in_pad.fill(114.0f); // Gray padding
    
    int dx = (target_size - scaled_w) / 2;
    int dy = (target_size - scaled_h) / 2;
    
    // Copy the resized image to the center of the padded image
    for (int y = 0; y < scaled_h; y++) {
        const float* src_row = in_resized.row(y);
        float* dst_row = in_pad.row(y + dy) + dx * 3;
        memcpy(dst_row, src_row, scaled_w * 3 * sizeof(float));
    }
    
    // Normalize
    const float mean_vals[3] = {0, 0, 0};
    const float norm_vals[3] = {1/255.0f, 1/255.0f, 1/255.0f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);
    
    // Create extractor from the model
    ncnn::Extractor ex = yoloNet.create_extractor();
    
    // Set input
    ex.input("images", in_pad);
    
    // Get output
    ncnn::Mat out;
    ex.extract("output", out);
    
    // Parse YOLOv5 detection output
    std::vector<Object> objects;
    
    // The output is a 25200 x 85 (80 classes + 5) matrix
    // (25200 = 3 * (80*80 + 40*40 + 20*20)) anchors
    // For each row: [x, y, w, h, confidence, 80 class probabilities]
    
    for (int i = 0; i < out.h; i++) {
        const float* values = out.row(i);
        
        float obj_conf = values[4];
        if (obj_conf < 0.25) { // Confidence threshold
            continue;
        }
        
        // Find the class with the highest probability
        float max_prob = 0.0f;
        int max_class = 0;
        for (int j = 0; j < 80; j++) {
            float class_prob = values[5 + j];
            if (class_prob > max_prob) {
                max_prob = class_prob;
                max_class = j;
            }
        }
        
        // Final confidence
        float confidence = obj_conf * max_prob;
        if (confidence < 0.25) { // Confidence threshold
            continue;
        }
        
        Object obj;
        obj.label = max_class;
        obj.prob = confidence;
        
        // The x,y,w,h are relative to the 640x640 input
        obj.x = values[0] / target_size;
        obj.y = values[1] / target_size;
        obj.w = values[2] / target_size;
        obj.h = values[3] / target_size;
        
        objects.push_back(obj);
    }
    
    // Apply NMS (Non-maximum suppression)
    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked, 0.45f); // 0.45 is the NMS threshold
    
    // Format results for Java
    std::vector<Object> results;
    for (int i : picked) {
        results.push_back(objects[i]);
    }
    
    // Create float array: [count, x, y, w, h, label, confidence, ...]
    int result_count = results.size();
    int float_array_size = 1 + result_count * 6; // 1 for count + 6 values per detection
    
    jfloatArray result = env->NewFloatArray(float_array_size);
    if (result == nullptr) {
        LOGE("Failed to create float array");
        return nullptr;
    }
    
    std::vector<float> resultData(float_array_size);
    resultData[0] = (float)result_count;
    
    for (int i = 0; i < result_count; i++) {
        const Object& obj = results[i];
        int offset = 1 + i * 6;
        resultData[offset + 0] = obj.x;
        resultData[offset + 1] = obj.y;
        resultData[offset + 2] = obj.w;
        resultData[offset + 3] = obj.h;
        resultData[offset + 4] = (float)obj.label;
        resultData[offset + 5] = obj.prob;
    }
    
    env->SetFloatArrayRegion(result, 0, float_array_size, resultData.data());
    
    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_releaseNative(JNIEnv* env, jobject /* this */) {
    // Clean up NCNN resources
    yoloNet.clear();
    modelLoaded = false;
    
    // Release Vulkan instance
    ncnn::destroy_gpu_instance();
    
    LOGI("Resources released");
}
