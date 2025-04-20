#include <jni.h>
#include <string>
#include <vector>
#include <algorithm> // For std::max, std::min, std::sort
#include <cmath>     // Use cmath instead of math.h for C++ style
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
static bool useGPU = false; // Whether Vulkan will be used
static int gpuDeviceIndex = -1; // Store selected GPU device index

// YOLOv11 constants (ADJUST THESE BASED ON YOUR SPECIFIC YOLOv11 MODEL)
const int YOLOV11_INPUT_WIDTH = 640;
const int YOLOV11_INPUT_HEIGHT = 640;
// Use thresholds consistent with Kotlin side or typical values
const float NMS_THRESHOLD = 0.45f;       // Non-Maximum Suppression threshold
const float CONFIDENCE_THRESHOLD = 0.25f; // Minimum confidence score to consider a detection
const int NUM_CLASSES = 84;              // Number of classes the model can detect (e.g., COCO dataset has 80, adjust if needed)

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
        if (areas[i] < 0) areas[i] = 0;
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
            else if (inter_area > 1e-6f) {
                 suppressed[j] = true;
            }
        }
    }
}

extern "C"
{

    // JNI Function: Initialize NCNN environment and check for GPU support.
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

        // Get AAssetManager - needed for model loading later, but good practice to get early
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        if (mgr == nullptr)
        {
            LOGE("Failed to get AAssetManager from Java context.");
            // Although not strictly needed for init itself, failure here suggests context issues.
            // Depending on app structure, might want to return false or proceed cautiously.
        }

        // Check Vulkan support
        int gpu_count = ncnn::get_gpu_count();
        if (gpu_count > 0)
        {
            // Select the default GPU (device 0)
            gpuDeviceIndex = 0; // Or implement logic to select a specific GPU if needed
            useGPU = true;
            LOGI("Vulkan GPU detected. Count: %d. Selected device index: %d. Enabling GPU acceleration.", gpu_count, gpuDeviceIndex);

            // Initialize Vulkan instance. This MUST happen before setting GPU options in the Net.
            // It's safe to call multiple times, but typically done once at init.
            ncnn::create_gpu_instance();
        }
        else
        {
            useGPU = false;
            gpuDeviceIndex = -1;
            LOGW("No Vulkan capable GPU detected or NCNN not built with Vulkan support. Using CPU.");
        }

        // Configure NCNN Net options for performance
        yoloNet.opt = ncnn::Option(); // Start with default options
        yoloNet.opt.lightmode = true;            // Enable light mode (reduces memory, might slightly impact speed)
        yoloNet.opt.num_threads = std::min(4, (int)std::thread::hardware_concurrency()); // Use up to 4 threads or max available if less
        yoloNet.opt.use_packing_layout = true;   // Highly recommended for performance on ARM CPU/GPU
        yoloNet.opt.use_fp16_packed = useGPU;    // Use FP16 packed format on GPU if available (good balance)
        yoloNet.opt.use_fp16_storage = useGPU;   // Store weights in FP16 on GPU if available (reduces memory)
        yoloNet.opt.use_fp16_arithmetic = false; // Use FP16 compute. Set to true ONLY if precision loss is acceptable and provides speedup. Often slower or similar speed but less precise. Test carefully. Defaulting to false is safer.
        yoloNet.opt.use_vulkan_compute = useGPU; // Explicitly enable Vulkan compute if GPU is available

        // Set the selected GPU device for the network instance
        if (useGPU) {
            yoloNet.set_vulkan_device(gpuDeviceIndex);
            LOGI("NCNN Net configured to use Vulkan device %d.", gpuDeviceIndex);
        } else {
            LOGI("NCNN Net configured to use CPU with %d threads.", yoloNet.opt.num_threads);
        }

        ncnnInitialized = true;
        modelLoaded = false; // Ensure model is marked as not loaded yet
        auto init_end_time = std::chrono::high_resolution_clock::now();
        auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end_time - init_start_time);
        LOGI("NCNN initialization complete (took %lld ms). Vulkan enabled: %s", init_duration.count(), useGPU ? "true" : "false");
        return JNI_TRUE;
    }

    // JNI Function: Load the YOLO model parameters and binary weights.
    JNIEXPORT jboolean JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_loadModel(JNIEnv *env, jobject /* this */, jobject assetManager)
    {
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

        // Get AAssetManager
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        if (mgr == nullptr)
        {
            LOGE("Failed to get AAssetManager from Java context for model loading.");
            return JNI_FALSE;
        }

        // *** ADJUST THESE FILENAMES TO MATCH YOUR YOLOv11 NCNN MODEL FILES ***
        // Ensure these files exist in your app's `src/main/assets` folder
        const char *param_filename = "yolov11.param"; // Example filename
        const char *bin_filename = "yolov11.bin";     // Example filename

        int ret_param = -1;
        int ret_bin = -1;
        {
            ncnn::MutexLockGuard guard(yoloNetLock); // Lock before modifying the global yoloNet
            // It's crucial to load both param and bin within the lock if yoloNet is shared

            // Load model parameters (.param file) using the asset manager
            // The second argument 'true' indicates path is relative within assets
            ret_param = yoloNet.load_param(mgr, param_filename);
            if (ret_param != 0)
            {
                LOGE("Failed to load model param file: %s (Error code: %d). Check if file exists in assets.", param_filename, ret_param);
                // No need to clear here, as nothing was loaded successfully yet.
                return JNI_FALSE;
            }
            LOGD("Loaded model param: %s", param_filename);

            // Load model weights (.bin file) using the asset manager
            ret_bin = yoloNet.load_model(mgr, bin_filename);
            if (ret_bin != 0)
            {
                LOGE("Failed to load model bin file: %s (Error code: %d). Check if file exists in assets.", bin_filename, ret_bin);
                // If bin loading fails after param loading succeeded, clear the net to be safe.
                yoloNet.clear(); // Clear partially loaded state
                return JNI_FALSE;
            }
            LOGD("Loaded model bin: %s", bin_filename);

        } // Mutex guard automatically released here

        modelLoaded = true;
        auto load_end_time = std::chrono::high_resolution_clock::now();
        auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end_time - load_start_time);
        LOGI("YOLOv11 model loading complete (took %lld ms). Model is ready for inference.", load_duration.count());
        return JNI_TRUE;
    }

    // JNI Function: Check if Vulkan (GPU) is initialized and being used.
    JNIEXPORT jboolean JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_hasVulkan(JNIEnv *env, jobject /* this */)
    {
        // Return the state determined during init and stored in global variables
        return (ncnnInitialized && useGPU) ? JNI_TRUE : JNI_FALSE;
    }

    // JNI Function: Perform object detection on an image provided as RGBA bytes.
    JNIEXPORT jfloatArray JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_detectNative(JNIEnv *env, jobject /* this */,
                                                                  jbyteArray imageBytes, jint imageWidth, jint imageHeight)
    {
        // --- Pre-checks ---
        if (!ncnnInitialized) {
            LOGE("Detection failed: NCNN not initialized.");
            return nullptr;
        }
        if (!modelLoaded) {
            LOGE("Detection failed: Model not loaded.");
            return nullptr;
        }
        if (imageBytes == nullptr || imageWidth <= 0 || imageHeight <= 0) {
            LOGE("Detection failed: Invalid image data provided (bytes=%p, width=%d, height=%d).", imageBytes, imageWidth, imageHeight);
            return nullptr;
        }

        // --- Overall Timing Start ---
        auto total_start_time = std::chrono::high_resolution_clock::now();

        // 1. Get image data from Java byte array
        //    ASSUMPTION: imageBytes contains raw RGBA pixel data (width * height * 4 bytes).
        jbyte *image_data_ptr = env->GetByteArrayElements(imageBytes, nullptr);
        if (image_data_ptr == nullptr) {
            LOGE("Failed to get image byte array elements. JNI error.");
            return nullptr;
        }
        // Cast to unsigned char* as expected by ncnn::Mat::from_pixels*
        const unsigned char *pixel_data = reinterpret_cast<const unsigned char *>(image_data_ptr);
        size_t expected_size = static_cast<size_t>(imageWidth) * imageHeight * 4; // RGBA = 4 bytes/pixel
        size_t actual_size = env->GetArrayLength(imageBytes);
        if (actual_size != expected_size) {
             LOGE("Image byte array size mismatch. Expected: %zu, Actual: %zu. Ensure RGBA format.", expected_size, actual_size);
             env->ReleaseByteArrayElements(imageBytes, image_data_ptr, JNI_ABORT); // Release buffer
             return nullptr;
        }


        // --- Preprocessing Timing Start ---
        auto preprocess_start_time = std::chrono::high_resolution_clock::now();

        // 2. Preprocessing: Create ncnn::Mat, resize with letterboxing, and normalize
        ncnn::Mat input_img;
        { // Scope for pixel_data usage before releasing the jbyteArray
            // Create ncnn::Mat directly from RGBA pixels.
            // ncnn::Mat::from_pixels_resize handles the resizing and letterboxing automatically.
            // It maintains aspect ratio and pads with black (0,0,0) by default.
            // Input: RGBA buffer, Pixel Type, Original Width, Original Height
            // Output: Target Width, Target Height (model input size)
            input_img = ncnn::Mat::from_pixels_resize(pixel_data, ncnn::Mat::PIXEL_RGBA2RGB, // Convert RGBA to RGB during load
                                                      imageWidth, imageHeight,
                                                      YOLOV11_INPUT_WIDTH, YOLOV11_INPUT_HEIGHT);

            // Release the Java byte array *immediately* after ncnn::Mat creation, as the data has been copied.
            // Use JNI_ABORT because we don't need to copy changes back (we only read).
            env->ReleaseByteArrayElements(imageBytes, image_data_ptr, JNI_ABORT);
            image_data_ptr = nullptr; // Avoid dangling pointer
            pixel_data = nullptr;
        } // End scope for pixel_data

        if (input_img.empty()) {
            LOGE("Failed to create or resize ncnn::Mat from input pixels.");
            return nullptr;
        }
        LOGD("Input image preprocessed to %d x %d x %d", input_img.w, input_img.h, input_img.c);

        // Normalize the image: Subtract mean and divide by normalization values.
        // For models trained on [0, 1] range, mean is {0, 0, 0} and norm is {1/255, 1/255, 1/255}.
        const float mean_vals[3] = {0.f, 0.f, 0.f};
        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        input_img.substract_mean_normalize(mean_vals, norm_vals);

        // --- Preprocessing Timing End ---
        auto preprocess_end_time = std::chrono::high_resolution_clock::now();
        auto preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end_time - preprocess_start_time);

        // 3. NCNN Inference
        ncnn::Mat out; // Output tensor from the network
        const char *input_name = nullptr; // Dynamically determined input tensor name
        const char *output_name = nullptr; // Dynamically determined output tensor name
        std::chrono::microseconds inference_duration(0); // Initialize duration

        { // Scope for extractor and mutex lock
            ncnn::MutexLockGuard guard(yoloNetLock); // Lock for thread safety during inference if yoloNet is shared
            ncnn::Extractor ex = yoloNet.create_extractor();
            // Optional: Configure extractor if needed (e.g., ex.set_num_threads(2))
            // Inherits options from yoloNet by default.

            // --- Inference Timing Start ---
            auto inference_start_time = std::chrono::high_resolution_clock::now();

            // Set input tensor. Try common names.
            // It's best practice to know the exact names from your model conversion process.
            input_name = "images"; // Common name for YOLO models
            int input_ret = ex.input(input_name, input_img);
            if (input_ret != 0) {
                 input_name = "in0"; // Another common name
                 LOGW("Failed input tensor name 'images', trying 'in0'");
                 input_ret = ex.input(input_name, input_img);
                 if (input_ret != 0) {
                    LOGE("Failed to set input tensor (tried 'images', 'in0'). Error: %d. Check your model's .param file for the correct input layer name.", input_ret);
                    // Consider retrieving input names dynamically if possible, though less common in NCNN examples
                    // std::vector<const char*> input_names = yoloNet.input_names();
                    return nullptr; // Critical failure
                 }
            }
            LOGD("Using input tensor name: %s", input_name);

            // Extract output tensor. Try common names.
            output_name = "output0"; // Common name, especially if generated by tools like Ultralytics export
            int extract_ret = ex.extract(output_name, out);
             if (extract_ret != 0) {
                output_name = "output"; // Another common name
                LOGW("Failed extract tensor name 'output0', trying 'output'");
                extract_ret = ex.extract(output_name, out);
                 if (extract_ret != 0) {
                    output_name = "out0"; // Yet another possibility
                    LOGW("Failed extract tensor name 'output', trying 'out0'");
                    extract_ret = ex.extract(output_name, out);
                     if (extract_ret != 0) {
                        LOGE("Failed to extract output tensor (tried 'output0', 'output', 'out0'). Error: %d. Check your model's .param file for the correct output layer name.", extract_ret);
                        // std::vector<const char*> output_names = yoloNet.output_names();
                        return nullptr; // Critical failure
                     }
                 }
            }
            LOGD("Using output tensor name: %s", output_name);

            // --- Inference Timing End ---
            auto inference_end_time = std::chrono::high_resolution_clock::now();
            inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end_time - inference_start_time);

        } // Mutex guard released, Extractor destroyed automatically

        // Check if output tensor is valid
        if (out.empty()) {
            LOGE("Output tensor '%s' is empty after extraction.", output_name);
            return nullptr;
        }

        // Log output tensor dimensions for debugging
        // NCNN tensor dimensions: w, h, d, c, dims
        LOGD("Output tensor ('%s') details: dims=%d, w=%d, h=%d, c=%d, total=%zu elements, elemsize=%zu bytes",
             output_name, out.dims, out.w, out.h, out.c, out.total(), out.elemsize);

        // --- Postprocessing Timing Start ---
        auto postprocess_start_time = std::chrono::high_resolution_clock::now();

        // 4. Postprocessing: Decode output tensor, apply NMS
        std::vector<Object> proposals; // Store valid detections before NMS

        // Calculate scaling factors to map detections from model input size (e.g., 640x640)
        // back to the original image size. This accounts for the letterboxing done by from_pixels_resize.
        float scale_w = (float)YOLOV11_INPUT_WIDTH / imageWidth;
        float scale_h = (float)YOLOV11_INPUT_HEIGHT / imageHeight;
        // The effective scale factor is the minimum of the two, due to aspect ratio preservation
        float scale = std::min(scale_w, scale_h);

        // Calculate the padding added during the resize operation
        // Padded width/height in the input_img coordinate system
        float scaled_img_w = imageWidth * scale;
        float scaled_img_h = imageHeight * scale;
        // Padding added to left/right and top/bottom
        float pad_left = (YOLOV11_INPUT_WIDTH - scaled_img_w) / 2.0f;
        float pad_top = (YOLOV11_INPUT_HEIGHT - scaled_img_h) / 2.0f;

        // --- Decode Output Tensor ---
        // YOLOv11 (and similar models like v8, v9, v10) often output in a [batch, features, num_detections] format.
        // Features usually are [cx, cy, w, h, class_score_0, class_score_1, ..., class_score_N-1].
        // NCNN might represent this as:
        // - dims=3: c=1, h=features, w=num_detections (if batch dim is squeezed)
        // - dims=2: h=features, w=num_detections (most likely after squeeze)
        // - dims=1: A flattened array (less common for this structure)
        // The provided code assumes a transposed format [features, num_detections] which is common after ONNX export/conversion.
        // Let's verify and adapt based on logged dimensions.

        int num_detections = 0;
        int num_features = 0; // Should be 4 (bbox) + NUM_CLASSES
        const float* output_data = nullptr;

        // Assuming the common [features, num_detections] format after potential squeeze
        if (out.dims == 2) { // Shape [features, num_detections]
             num_features = out.h;
             num_detections = out.w;
             output_data = (const float*)out.data; // Direct access to contiguous data
             LOGD("Parsing output format: [features=%d, num_detections=%d]", num_features, num_detections);
        }
        // Handle potential [1, features, num_detections] case (batch dim not squeezed)
        else if (out.dims == 3 && out.c == 1) {
             num_features = out.h;
             num_detections = out.w;
             // Access data for the first (and only) batch item
             // For NCNN Mat with dims=3, data is often accessed via channel(c).row(h)
             // Assuming c=1, h=features, w=detections. Access the start of the data.
             output_data = out.channel(0).row(0); // Get pointer to the start of the data for batch 0
             LOGD("Parsing output format: [batch=1, features=%d, num_detections=%d]", num_features, num_detections);
        }
        else {
            LOGE("Unsupported output tensor shape for YOLOv11 postprocessing. Dims=%d, W=%d, H=%d, C=%d. Expected dims=2 or dims=3 with C=1.", out.dims, out.w, out.h, out.c);
            return nullptr; // Indicate error
        }

        // Validate dimensions
        int expected_features = 4 + NUM_CLASSES; // e.g., 4 + 84 = 88
        if (num_features != expected_features) {
            LOGE("Output tensor feature count mismatch! Expected: %d (4 bbox + %d classes), Got: %d. Check model structure or NUM_CLASSES constant.",
                 expected_features, NUM_CLASSES, num_features);
            return nullptr; // Indicate error
        }
        if (num_detections <= 0 || output_data == nullptr) {
            LOGW("Output tensor indicates zero potential detections (%d) or data pointer is null. Returning empty result.", num_detections);
            // Return an array containing only the count (0.0f)
            jfloatArray emptyResult = env->NewFloatArray(1);
            if (emptyResult) {
                float zero = 0.0f;
                env->SetFloatArrayRegion(emptyResult, 0, 1, &zero);
            } else {
                LOGE("Failed to allocate empty result array.");
            }
            return emptyResult;
        }

        // --- Extract Boxes and Scores ---
        std::vector<Object> raw_objects;
        raw_objects.reserve(num_detections / 2); // Pre-allocate reasonable space

        // Iterate through each potential detection COLUMN (index `i`)
        // Data layout: output_data[feature_row * num_detections + detection_col]
        for (int i = 0; i < num_detections; ++i) {
            // Find the class with the highest score for this detection `i`
            int best_class_idx = -1;
            float max_class_prob = -1.0f;

            // Class scores start at feature index 4
            const float* class_scores_ptr = output_data + 4 * num_detections; // Pointer to the start of class scores for all detections

            for (int c = 0; c < NUM_CLASSES; ++c) {
                // Access score for class 'c', detection 'i'
                // float class_prob = output_data[(c + 4) * num_detections + i]; // Original indexing
                float class_prob = class_scores_ptr[c * num_detections + i]; // Optimized access
                if (class_prob > max_class_prob) {
                    max_class_prob = class_prob;
                    best_class_idx = c;
                }
            }

            // Filter by confidence threshold
            if (max_class_prob >= CONFIDENCE_THRESHOLD) {
                // Extract bounding box coordinates (center_x, center_y, width, height) for detection 'i'
                // Bbox data is at feature rows 0, 1, 2, 3
                float cx = output_data[0 * num_detections + i]; // center_x
                float cy = output_data[1 * num_detections + i]; // center_y
                float w  = output_data[2 * num_detections + i]; // width
                float h  = output_data[3 * num_detections + i]; // height

                // Convert box from network input coords (e.g., 640x640) to original image coords
                // 1. Convert center (cx, cy) to top-left (x1_net, y1_net) in network coords
                float x1_net = cx - w / 2.0f;
                float y1_net = cy - h / 2.0f;
                // Note: width and height (w, h) are already in network coords

                // 2. Remove padding and scale back to original image dimensions
                float x1_orig = (x1_net - pad_left) / scale;
                float y1_orig = (y1_net - pad_top) / scale;
                float w_orig = w / scale;
                float h_orig = h / scale;

                // 3. Clamp coordinates to be within the original image boundaries [0, width-1] and [0, height-1]
                // Also ensure width/height are positive after potential floating point inaccuracies.
                float x2_orig = x1_orig + w_orig;
                float y2_orig = y1_orig + h_orig;

                x1_orig = std::max(0.0f, std::min((float)imageWidth - 1.0f, x1_orig));
                y1_orig = std::max(0.0f, std::min((float)imageHeight - 1.0f, y1_orig));
                x2_orig = std::max(x1_orig, std::min((float)imageWidth - 1.0f, x2_orig)); // Ensure x2 >= x1
                y2_orig = std::max(y1_orig, std::min((float)imageHeight - 1.0f, y2_orig)); // Ensure y2 >= y1

                // Recalculate width/height after clamping
                w_orig = x2_orig - x1_orig;
                h_orig = y2_orig - y1_orig;

                // Only add the object if it has a valid size after clamping and meets confidence threshold
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
            }
        }
        LOGD("Extracted %zu raw boxes above confidence threshold %.2f.", raw_objects.size(), CONFIDENCE_THRESHOLD);

        // --- Non-Maximum Suppression (NMS) ---
        // Sort boxes by confidence score (descending) before applying NMS
        std::sort(raw_objects.begin(), raw_objects.end(), [](const Object &a, const Object &b) {
            return a.prob > b.prob;
        });

        // Perform NMS to remove overlapping boxes
        std::vector<int> picked_indices;
        nms_sorted_bboxes(raw_objects, picked_indices, NMS_THRESHOLD);
        LOGD("NMS filtering with threshold %.2f resulted in %zu final boxes.", NMS_THRESHOLD, picked_indices.size());

        // Collect final proposals based on NMS results
        proposals.reserve(picked_indices.size());
        for (int index : picked_indices) {
            if (index >= 0 && static_cast<size_t>(index) < raw_objects.size()) { // Bounds check
                 proposals.push_back(raw_objects[index]);
            } else {
                 LOGW("Invalid index %d from NMS, skipping.", index);
            }
        }

        // --- Postprocessing Timing End ---
        auto postprocess_end_time = std::chrono::high_resolution_clock::now();
        auto postprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end_time - postprocess_start_time);

        // 5. Format results for Java/Kotlin
        // Output format: [count, x1_1, y1_1, w_1, h_1, label_1, score_1, x1_2, y1_2, ...]
        int final_count = proposals.size();
        int result_elements = 1 + final_count * 6; // 1 for count, 6 floats per detection
        jfloatArray resultArray = env->NewFloatArray(result_elements);
        if (resultArray == nullptr) {
            LOGE("Failed to allocate float array for JNI results (size %d).", result_elements);
            return nullptr; // Allocation failed
        }

        // Create a temporary buffer on the C++ side
        std::vector<float> resultData(result_elements);
        resultData[0] = static_cast<float>(final_count); // First element is the count

        // Fill the buffer with detection data
        for (int i = 0; i < final_count; ++i) {
            const Object &obj = proposals[i];
            int offset = 1 + i * 6; // Start index for this object's data
            resultData[offset + 0] = obj.x;
            resultData[offset + 1] = obj.y;
            resultData[offset + 2] = obj.w;
            resultData[offset + 3] = obj.h;
            resultData[offset + 4] = static_cast<float>(obj.label); // Class index
            resultData[offset + 5] = obj.prob;                      // Confidence score
        }

        // Copy the C++ buffer data to the Java float array
        env->SetFloatArrayRegion(resultArray, 0, result_elements, resultData.data());

        // --- Overall Timing End ---
        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);

        LOGI("Detection complete. Timing (us): Total=%lld | Preproc=%lld | Infer=%lld | Postproc=%lld | Objects=%d",
             total_duration_us.count(),
             preprocess_duration.count(),
             inference_duration.count(),
             postprocess_duration.count(),
             final_count);

        return resultArray;
    }

    // JNI Function: Release NCNN resources, including network and GPU context.
    JNIEXPORT void JNICALL
    Java_com_example_yolo_1kotlin_1ncnn_NcnnDetector_releaseNative(JNIEnv *env, jobject /* this */)
    {
        LOGI("Releasing NCNN resources...");
        {
            ncnn::MutexLockGuard guard(yoloNetLock); // Ensure thread safety during cleanup
            if (modelLoaded) {
                yoloNet.clear(); // Clear the network (releases model data and internal context)
                LOGD("NCNN Net cleared.");
            } else {
                LOGD("NCNN Net was not loaded, skipping clear.");
            }
        } // Mutex released

        // Destroy the Vulkan instance if it was created
        if (useGPU)
        {
            // This should be called only once when the application is shutting down
            // or when NCNN GPU usage is definitively finished.
            ncnn::destroy_gpu_instance();
            LOGI("Vulkan GPU instance destroyed.");
        } else {
             LOGI("No Vulkan GPU instance to destroy (was using CPU).");
        }

        // Reset state variables
        ncnnInitialized = false;
        modelLoaded = false;
        useGPU = false;
        gpuDeviceIndex = -1;
        LOGI("NCNN resources released and state reset.");
    }

} // extern "C"
