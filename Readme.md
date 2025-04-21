# YOLOv11 Object Detection on Android with NCNN and Kotlin/Compose

This project demonstrates real-time object detection on Android using a YOLOv11 model accelerated by the NCNN library. The application is built with Kotlin and Jetpack Compose for the UI, leveraging CameraX for camera input and JNI for communication with the native NCNN inference engine.

## Overview

The application captures frames from the device camera using CameraX, processes them in YUV format, and sends the relevant data (Y, U, V planes, strides, dimensions) to a native C++ layer via JNI. The C++ layer utilizes NCNN to:

1.  Convert the YUV frame to RGB format.
2.  Preprocess the RGB image (resize, normalize) to match the YOLOv11 model's input requirements (640x640).
3.  Perform inference using the loaded YOLOv11 model (potentially accelerated by Vulkan GPU if available).
4.  Post-process the model's output, applying confidence thresholding and Non-Maximum Suppression (NMS) to filter detections.
5.  Return the final detection results (bounding boxes, class labels, confidence scores) back to the Kotlin layer.

The Kotlin layer then uses Jetpack Compose to draw the camera preview and overlay the detected bounding boxes with labels and confidence scores onto the screen.

## Features

*   **Real-time Object Detection:** Utilizes the YOLOv11 model via NCNN for efficient detection.
*   **NCNN Integration:** Leverages the NCNN deep learning framework for high-performance inference on mobile devices.
*   **CPU/GPU Acceleration:** Automatically detects and utilizes Vulkan-capable GPUs for faster inference, falling back to CPU otherwise.
*   **CameraX:** Uses Android's modern CameraX library for efficient camera stream handling.
*   **Jetpack Compose UI:** Modern UI built with Jetpack Compose, including a custom `Canvas` for drawing bounding boxes.
*   **YUV Processing:** Handles YUV_420_888 image format directly from CameraX, performing conversion to RGB natively.
*   **JNI Communication:** Efficient communication between Kotlin and C++ using the Java Native Interface.
*   **Dynamic Bounding Box Overlay:** Renders bounding boxes scaled correctly over the camera preview, adapting to different screen and camera resolutions.

## Requirements

*   Android Studio (latest recommended version)
*   Android SDK (check `app/build.gradle.kts` for specific `minSdk` and `compileSdk`)
*   Android NDK (for building the C++ code)
*   **YOLOv11 NCNN Model Files:** You need to obtain the `.param` and `.bin` files for your specific YOLOv11 model trained for NCNN. Place these files in the `app/src/main/assets/` directory.
    *   The current C++ code expects files named `yolov11.param` and `yolov11.bin`. Update `native-lib.cpp` if your filenames differ.
*   **Class Names:** Ensure the `cocoClassNames` list in `BoundingBoxOverlay.kt` matches the classes your YOLOv11 model was trained on. The current list includes standard COCO classes plus custom ones ("pump", "pipe", "steel pipe", "electric cable"), totaling 84 classes. Adjust `NUM_CLASSES` in `native-lib.cpp` accordingly if needed.

## Setup and Build

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd yolo_kotlin_ncnn
    ```
2.  **Place Model Files:** Copy your `yolov11.param` and `yolov11.bin` files into the `app/src/main/assets/` directory.
3.  **Open in Android Studio:** Open the project root directory in Android Studio.
4.  **Sync Gradle:** Let Android Studio sync the project and download dependencies.
5.  **Build:** Build the project (Build > Make Project). This will compile the Kotlin code and the C++ native library using CMake and the NDK.
6.  **Run:** Run the application on a connected Android device or emulator.

## Usage

*   Grant camera permissions when prompted.
*   The app will display the camera preview.
*   Detected objects will be highlighted with bounding boxes, class labels, and confidence scores.
*   Logs in Logcat (tags: `MainActivity`, `NcnnDetector`, `BoundingBoxOverlay`, `NCNN_YOLOv11_Native`) provide detailed information about initialization, model loading, inference times, and potential errors.

## Technical Details

*   **NCNN Initialization:** The `NcnnDetector` class handles loading the `native-lib` shared library and initializing the NCNN environment (`initNative`). It checks for Vulkan support (`hasVulkan`) and configures NCNN options for optimal performance (light mode, threading, FP16 usage on GPU).
*   **Model Loading:** The `.param` and `.bin` files are loaded from the assets directory using the Android `AssetManager` passed to the native layer (`loadModel`).
*   **YUV to RGB Conversion:** The `detectNative` function in C++ receives direct `ByteBuffer` pointers for Y, U, and V planes. It performs manual YUV_420_888 (handling both planar I420/YV12 and semi-planar NV21 based on `uvPixelStride`) to RGB conversion before creating the `ncnn::Mat`.
*   **Preprocessing:** The RGB `ncnn::Mat` is resized (with letterboxing/padding) to the model's input size (640x640) and normalized (pixels divided by 255.0).
*   **Inference:** The preprocessed `ncnn::Mat` is fed into the `yoloNet` extractor. Input and output tensor names (`in0`, `out0`) are hardcoded based on the expected NCNN model structure.
*   **Postprocessing:** The raw output tensor from NCNN is processed:
    *   Scores for each class are checked against `CONFIDENCE_THRESHOLD`.
    *   Bounding box coordinates (center x/y, width/height) are decoded and scaled back to the original image dimensions, accounting for padding.
    *   Non-Maximum Suppression (NMS) is applied using `NMS_THRESHOLD` to remove overlapping boxes for the same object.
*   **Bounding Box Rendering:** The `BoundingBoxOverlay` Composable takes the list of `Detection` objects. It calculates the correct scaling and offset to map the detection coordinates (from the original camera frame size) onto the `Canvas` coordinates, ensuring the boxes align with the preview, assuming a `FIT_CENTER` scale type for the preview.

## License

AGPL-3.0 License. 
