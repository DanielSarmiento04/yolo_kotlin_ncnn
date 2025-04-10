#include <jni.h>
#include <android/log.h>
#include "ncnn/net.h"  // Include the NCNN headers from your ncnn_include folder

// Create an instance of the NCNN Net that you'll use for inference.
static ncnn::Net net;

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_initNative(JNIEnv* env, jobject /* this */, jobject assetManager) {
    // Log to verify that the native method is called
    __android_log_print(ANDROID_LOG_INFO, "NCNN", "Native init called");
    
    // Configure NCNN options to explicitly use CPU-only mode
    net.opt.num_threads = 4;  // Use multiple threads for CPU computation
    net.opt.lightmode = true; // Enable light mode for faster inference
    
    // Always ensure Vulkan is disabled regardless of how NCNN was compiled
    #ifdef NCNN_VULKAN
    net.opt.use_vulkan_compute = false;
    __android_log_print(ANDROID_LOG_INFO, "NCNN", "Explicitly disabled Vulkan in runtime");
    #endif
    
    // Return success
    return JNI_TRUE;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_releaseNative(JNIEnv* env, jobject /* this */) {
    // Clean up NCNN resources
    net.clear();
}
