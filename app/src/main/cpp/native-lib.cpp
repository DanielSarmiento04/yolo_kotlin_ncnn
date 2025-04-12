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
    
    // Implementation for detection would go here
    // This would involve:
    // 1. Converting imageBytes to ncnn::Mat
    // 2. Processing with YOLO model
    // 3. Parsing results
    // 4. Returning detection array
    
    // Placeholder return
    jfloatArray result = env->NewFloatArray(1);
    jfloat fill[1] = {0.0f};
    env->SetFloatArrayRegion(result, 0, 1, fill);
    
    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_releaseNative(JNIEnv* env, jobject /* this */) {
    // Clean up NCNN resources
    yoloNet.clear();
    modelLoaded = false;
    ncnn::destroy_gpu_instance();
    LOGI("Resources released");
}
