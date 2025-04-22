package com.example.yolo_kotlin_ncnn

import android.content.Context
import android.content.res.AssetManager
import android.graphics.RectF
import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer

/**
 * Represents a single detected object.
 *
 * @param label The index of the detected class label.
 * @param confidence The confidence score of the detection (0.0 to 1.0).
 * @param rect The bounding box of the detected object in original image coordinates.
 */
data class Detection(
    val label: Int,
    val confidence: Float,
    val rect: RectF
)

/**
 * Manages the NCNN YOLOv11 detector instance and interactions via JNI.
 *
 * This class handles initialization, model loading, running inference asynchronously,
 * and releasing native resources.
 *
 * @param context The application context, used to access assets.
 * @param dispatcher The coroutine dispatcher for running inference (defaults to IO).
 */
class NcnnDetector(
    context: Context,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) {
    private val assetManager: AssetManager = context.assets
    // Make status flags publicly readable
    var isInitialized: Boolean = false
        private set // Allow reading from outside, but only set internally
    var isModelLoaded: Boolean = false
        private set
    var hasVulkanGpu: Boolean = false
        private set

    init {
        try {
            System.loadLibrary("native-lib")
            Log.i(TAG, "Native library loaded successfully.")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load native library: ${e.message}")
            // Handle library loading failure (e.g., show error to user)
        }
    }

    /**
     * Initializes the native NCNN environment.
     * Must be called before loading the model or running detection.
     * Checks for Vulkan support.
     *
     * @return True if initialization was successful, false otherwise.
     */
    suspend fun init(): Boolean = withContext(dispatcher) {
        if (isInitialized) {
            Log.i(TAG, "NCNN already initialized.")
            return@withContext true
        }
        isInitialized = initNative(assetManager)
        if (isInitialized) {
            hasVulkanGpu = hasVulkan() // Check Vulkan status via JNI
            Log.i(TAG, "NCNN initialized successfully. Vulkan GPU available: $hasVulkanGpu")
        } else {
            Log.e(TAG, "NCNN initialization failed.")
            hasVulkanGpu = false // Ensure flag is false on failure
        }
        isInitialized
    }

    /**
     * Loads the YOLOv11 model from assets.
     * Requires `init()` to have been called successfully.
     *
     * @return True if the model was loaded successfully, false otherwise.
     */
    suspend fun loadModel(): Boolean = withContext(dispatcher) {
        if (!isInitialized) {
            Log.e(TAG, "Cannot load model: NCNN not initialized.")
            return@withContext false
        }
        if (isModelLoaded) {
            Log.i(TAG, "Model already loaded.")
            return@withContext true
        }
        isModelLoaded = loadModelNative(assetManager)
        if (isModelLoaded) {
            Log.i(TAG, "YOLOv11 model loaded successfully.")
        } else {
            Log.e(TAG, "Failed to load YOLOv11 model.")
        }
        isModelLoaded
    }

    /**
     * Checks if Vulkan GPU is being used by the detector.
     * Requires `init()` to have been called.
     * This now directly returns the internal flag set during init.
     *
     * @return True if Vulkan is initialized and used, false otherwise.
     */
    fun isVulkanSupported(): Boolean {
        // Return the flag set during init()
        return isInitialized && hasVulkanGpu
    }

    /**
     * Performs object detection on the provided YUV image data asynchronously.
     * Requires `init()` and `loadModel()` to have been called successfully.
     *
     * @param yBuffer ByteBuffer for the Y plane.
     * @param uBuffer ByteBuffer for the U plane.
     * @param vBuffer ByteBuffer for the V plane.
     * @param yStride Row stride for the Y plane.
     * @param uvStride Row stride for the U/V planes.
     * @param uvPixelStride Pixel stride for the U/V planes.
     * @param width The width of the input image.
     * @param height The height of the input image.
     * @return A list of `Detection` objects, or null if detection failed or input is invalid.
     */
    suspend fun detect(
        yBuffer: ByteBuffer?, uBuffer: ByteBuffer?, vBuffer: ByteBuffer?,
        yStride: Int, uvStride: Int, uvPixelStride: Int,
        width: Int, height: Int
    ): List<Detection>? = withContext(dispatcher) {
        if (!isInitialized || !isModelLoaded) {
            Log.e(TAG, "Cannot detect: NCNN not initialized or model not loaded.")
            return@withContext null
        }
        if (yBuffer == null || !yBuffer.isDirect ||
            uBuffer == null || !uBuffer.isDirect ||
            vBuffer == null || !vBuffer.isDirect ||
            width <= 0 || height <= 0 || yStride <= 0 || uvStride <= 0 || uvPixelStride <= 0) {
            Log.e(TAG, "Cannot detect: Invalid or non-direct YUV input data/params.")
            return@withContext null
        }

        // 1. Call native detect function with YUV data
        val startTime = System.currentTimeMillis()
        val results: FloatArray? = detectNative(
            yBuffer, uBuffer, vBuffer,
            yStride, uvStride, uvPixelStride,
            width, height
        )
        val endTime = System.currentTimeMillis()
        Log.d(TAG, "Native YUV detection call took: ${endTime - startTime} ms")

        // 2. Parse results
        parseDetectionResults(results)
    }

    /**
     * Parses the float array returned by the native `detectNative` function.
     * Format: [count, x1_1, y1_1, w_1, h_1, label_1, score_1, x1_2, y1_2, ...]
     */
    private fun parseDetectionResults(results: FloatArray?): List<Detection>? {
        if (results == null) {
            Log.e(TAG, "Detection failed: Native function returned null.")
            return null
        }
        if (results.isEmpty()) {
            Log.w(TAG, "Detection returned empty results array.")
            return emptyList()
        }

        val count = results[0].toInt()
        if (count < 0) {
            Log.e(TAG, "Detection failed: Native function returned negative count ($count).")
            return null
        }
        if (count == 0) {
            return emptyList()
        }

        val expectedSize = 1 + count * 6
        if (results.size != expectedSize) {
            Log.e(TAG, "Result array size mismatch. Expected: $expectedSize, Got: ${results.size}")
            return null
        }

        val detections = mutableListOf<Detection>()
        for (i in 0 until count) {
            val offset = 1 + i * 6
            try {
                val x = results[offset + 0]
                val y = results[offset + 1]
                val w = results[offset + 2]
                val h = results[offset + 3]
                val label = results[offset + 4].toInt()
                val score = results[offset + 5]

                if (w <= 0f || h <= 0f || label < 0 || score < 0f || score > 1f) {
                    Log.w(TAG, "Skipping invalid detection data at index $i: [x=$x, y=$y, w=$w, h=$h, label=$label, score=$score]")
                    continue
                }

                detections.add(
                    Detection(
                        label = label,
                        confidence = score,
                        rect = RectF(x, y, x + w, y + h)
                    )
                )
            } catch (e: ArrayIndexOutOfBoundsException) {
                Log.e(TAG, "Error parsing detection results at index $i. Array index out of bounds.", e)
                return null
            }
        }
        Log.i(TAG, "Successfully parsed $count detections.")
        return detections
    }

    /**
     * Releases the native NCNN resources.
     * Should be called when the detector is no longer needed (e.g., in onDestroy).
     */
    fun release() {
        if (isInitialized) { // Check the flag before calling native
            releaseNative()
            Log.i(TAG, "NCNN resources released via native call.")
            // Reset flags after successful release
            isInitialized = false
            isModelLoaded = false
            hasVulkanGpu = false
        } else {
            Log.i(TAG, "NCNN resources already released or never initialized.")
        }
    }

    // --- Native Method Declarations ---

    /** Initializes NCNN environment. */
    private external fun initNative(assetManager: AssetManager): Boolean

    /** Loads the NCNN model files. */
    private external fun loadModelNative(assetManager: AssetManager): Boolean

    /** Checks if Vulkan GPU is being used. */
    private external fun hasVulkan(): Boolean

    /** Performs detection on the YUV image data. */
    private external fun detectNative(
        yBuffer: ByteBuffer, uBuffer: ByteBuffer, vBuffer: ByteBuffer,
        yStride: Int, uvStride: Int, uvPixelStride: Int,
        imageWidth: Int, imageHeight: Int
    ): FloatArray?

    /** Releases NCNN resources. */
    private external fun releaseNative()

    companion object {
        private const val TAG = "NcnnDetector"
    }
}
