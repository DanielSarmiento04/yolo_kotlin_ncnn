package com.example.yolo_kotlin_ncnn

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import java.nio.ByteBuffer
import android.util.Log

class NcnnDetector(private val context: Context) {

    // Data class to hold detection results
    data class Detection(
        val rect: RectF,    // Bounding box in original image coordinates
        val label: Int,     // Class index
        val confidence: Float // Confidence score
    )

    companion object {
        private const val TAG = "NcnnDetector"

        // Load the native library compiled by CMake
        init {
            try {
                System.loadLibrary("native-lib") // Ensure this matches your CMakeLists.txt target name
                Log.i(TAG, "Successfully loaded native-lib")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native-lib", e)
                // Handle error appropriately, maybe show a message to the user
            }
        }
    }

    // Native method declarations matching C++ JNI functions
    // Note: AssetManager is passed directly from Kotlin
    private external fun initNative(assetManager: AssetManager): Boolean
    private external fun loadModel(assetManager: AssetManager): Boolean
    private external fun hasVulkan(): Boolean // Checks if Vulkan is being used by NCNN
    private external fun detect(imageBytes: ByteArray, width: Int, height: Int): FloatArray? // Returns nullable FloatArray
    private external fun releaseNative()

    // State variables managed by the Kotlin wrapper
    private var isInitialized = false
    private var isModelLoaded = false
    private var isVulkanAvailable = false // Store Vulkan status after init

    // Initialize NCNN. Call this first.
    fun init(): Boolean {
        if (isInitialized) return true

        // Pass the application's asset manager to native code
        isInitialized = try {
            // Pass context.assets which is the AssetManager
            initNative(context.assets)
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Native init method not found", e)
            false
        }

        if (isInitialized) {
            isVulkanAvailable = hasVulkan() // Check Vulkan status after successful init
            Log.i(TAG, "NCNN initialized successfully. Vulkan available: $isVulkanAvailable")
        } else {
            Log.e(TAG, "NCNN initialization failed")
        }
        return isInitialized
    }

    // Load the YOLO model. Requires init() to be called first.
    fun loadModel(): Boolean {
        if (!isInitialized) {
            Log.e(TAG, "Cannot load model: NCNN not initialized.")
            return false
        }
        if (isModelLoaded) return true

        isModelLoaded = try {
            // Pass context.assets which is the AssetManager
            loadModel(context.assets)
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Native loadModel method not found", e)
            false
        }

        if (isModelLoaded) {
            Log.i(TAG, "YOLO model loaded successfully")
        } else {
            Log.e(TAG, "Failed to load YOLO model")
        }
        return isModelLoaded
    }

    /**
     * Detect objects in an RGBA ByteArray.
     *
     * @param rgbaBytes The input image pixel data in RGBA format.
     * @param width The width of the image.
     * @param height The height of the image.
     * @param confidenceThreshold Minimum confidence score for a detection to be included in results.
     * @return A list of Detection objects. Returns empty list if detection fails or no objects are found.
     */
    fun detect(rgbaBytes: ByteArray, width: Int, height: Int, confidenceThreshold: Float = 0.5f): List<Detection> {
        if (!isInitialized || !isModelLoaded) {
            Log.e(TAG, "Cannot detect: NCNN not initialized or model not loaded.")
            return emptyList()
        }

        // 1. Validate input
        val expectedSize = width * height * 4
        if (rgbaBytes.size != expectedSize) {
            Log.e(TAG, "Input byte array size (${rgbaBytes.size}) does not match expected size ($expectedSize) for ${width}x${height} RGBA image.")
            return emptyList()
        }

        // 2. Call native detect function (already expects ByteArray, width, height)
        val startTime = System.currentTimeMillis()
        val rawResult: FloatArray? = try {
            // Pass the RGBA byte array directly
            detect(rgbaBytes, width, height)
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Native detect method not found", e)
            null
        } catch (e: Exception) {
            Log.e(TAG, "Error during native detection", e)
            null
        }
        val endTime = System.currentTimeMillis()
        // Log detection time less frequently or only if > threshold to reduce spam
        if (endTime - startTime > 10) { // Example: Log only if detection takes > 10ms
             Log.i(TAG, "Detection time: ${endTime - startTime} ms")
        }

        // 3. Parse results (This part remains the same)
        if (rawResult == null || rawResult.isEmpty()) {
            // Log only if null, empty array with count 0 is valid (no detections)
            if (rawResult == null) Log.w(TAG, "Detection returned null result.")
            else Log.i(TAG, "Detection returned empty or invalid array.") // Log if empty array received
            return emptyList()
        }
        // *** ADDED LOGGING ***
        Log.d(TAG, "Raw result array (first 10 elements): ${rawResult.take(10).joinToString()}")
        // *** END ADDED LOGGING ***

        // Format: [count, x1, y1, w1, h1, label1, conf1, x2, y2, w2, h2, label2, conf2, ...]
        val count = rawResult[0].toInt()
        // *** ADDED LOGGING ***
        Log.d(TAG, "Parsed detection count from native: $count")
        // *** END ADDED LOGGING ***
        if (count <= 0) {
            // Log.i(TAG, "No objects detected.") // Less verbose logging for no detections
            return emptyList()
        }

        val detections = mutableListOf<Detection>()
        // Expected size check
        val expectedSizeResult = 1 + count * 6 // Renamed variable to avoid conflict
        if (rawResult.size < expectedSizeResult) {
             Log.e(TAG, "Result array size (${rawResult.size}) is smaller than expected ($expectedSizeResult) for $count detections.")
             return emptyList() // Avoid IndexOutOfBoundsException
        }

        for (i in 0 until count) {
            val offset = 1 + i * 6
            val x = rawResult[offset + 0]
            val y = rawResult[offset + 1]
            val w = rawResult[offset + 2]
            val h = rawResult[offset + 3]
            val label = rawResult[offset + 4].toInt()
            val confidence = rawResult[offset + 5]

            // Apply confidence threshold here (optional, can also be done in native code)
            if (confidence >= confidenceThreshold) {
                // Create RectF from top-left coordinates and width/height
                // Ensure coordinates are valid before creating RectF
                 if (x >= 0 && y >= 0 && w > 0 && h > 0) {
                    val rect = RectF(x, y, x + w, y + h)
                    detections.add(Detection(rect, label, confidence))
                 } else {
                    Log.w(TAG, "Skipping detection with invalid coordinates: x=$x, y=$y, w=$w, h=$h")
                 }
            }
        }
        if (detections.isNotEmpty()) {
            Log.i(TAG, "Detected ${detections.size} objects (after thresholding).")
        } else if (count > 0) {
             // Log if native reported count > 0 but Kotlin list is empty (due to thresholding or invalid data)
             Log.i(TAG, "Detected 0 objects after Kotlin thresholding/validation (native count was $count).")
        }
        return detections
    }

    /**
     * Detect objects in a Bitmap.
     *
     * @param bitmap The input image. IMPORTANT: Must be in ARGB_8888 format for correct byte conversion.
     * @param confidenceThreshold Minimum confidence score for a detection to be included in results.
     * @return A list of Detection objects. Returns empty list if detection fails or no objects are found.
     */
    fun detect(bitmap: Bitmap, confidenceThreshold: Float = 0.5f): List<Detection> {
        if (!isInitialized || !isModelLoaded) {
            Log.e(TAG, "Cannot detect: NCNN not initialized or model not loaded.")
            return emptyList()
        }

        // 1. Ensure Bitmap is in ARGB_8888 format for reliable byte extraction
        val argbBitmap = if (bitmap.config == Bitmap.Config.ARGB_8888) {
            bitmap
        } else {
            Log.w(TAG, "Input bitmap is not ARGB_8888, converting...")
            bitmap.copy(Bitmap.Config.ARGB_8888, true)
        }

        // 2. Convert Bitmap to RGBA byte array
        // NCNN native code expects RGBA (as implemented in native-lib.cpp).
        val bytes = ByteArray(argbBitmap.byteCount)
        val buffer = ByteBuffer.wrap(bytes)
        argbBitmap.copyPixelsToBuffer(buffer)
        // The buffer now contains RGBA data if the bitmap was ARGB_8888.

        // 3. Call native detect function
        val startTime = System.currentTimeMillis()
        val rawResult: FloatArray? = try {
            detect(bytes, argbBitmap.width, argbBitmap.height)
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Native detect method not found", e)
            null
        } catch (e: Exception) {
            Log.e(TAG, "Error during native detection", e)
            null
        }
        val endTime = System.currentTimeMillis()
        Log.i(TAG, "Detection time: ${endTime - startTime} ms")


        // 4. Parse results
        if (rawResult == null || rawResult.isEmpty()) {
            // Log only if null, empty array with count 0 is valid (no detections)
            if (rawResult == null) Log.w(TAG, "Detection returned null result.")
            else Log.i(TAG, "Detection returned empty or invalid array.") // Log if empty array received
            return emptyList()
        }

        // *** ADDED LOGGING ***
        Log.d(TAG, "Raw result array (first 10 elements): ${rawResult.take(10).joinToString()}")
        // *** END ADDED LOGGING ***

        // Format: [count, x1, y1, w1, h1, label1, conf1, x2, y2, w2, h2, label2, conf2, ...]
        val count = rawResult[0].toInt()
        // *** ADDED LOGGING ***
        Log.d(TAG, "Parsed detection count from native: $count")
        // *** END ADDED LOGGING ***
        if (count <= 0) {
            // Log.i(TAG, "No objects detected.") // Less verbose logging for no detections
            return emptyList()
        }

        val detections = mutableListOf<Detection>()
        // Expected size check
        val expectedSize = 1 + count * 6
        if (rawResult.size < expectedSize) {
             Log.e(TAG, "Result array size (${rawResult.size}) is smaller than expected ($expectedSize) for $count detections.")
             return emptyList() // Avoid IndexOutOfBoundsException
        }


        for (i in 0 until count) {
            val offset = 1 + i * 6
            val x = rawResult[offset + 0]
            val y = rawResult[offset + 1]
            val w = rawResult[offset + 2]
            val h = rawResult[offset + 3]
            val label = rawResult[offset + 4].toInt()
            val confidence = rawResult[offset + 5]

            // Apply confidence threshold here (optional, can also be done in native code)
            if (confidence >= confidenceThreshold) {
                // Create RectF from top-left coordinates and width/height
                // Ensure coordinates are valid before creating RectF
                 if (x >= 0 && y >= 0 && w > 0 && h > 0) {
                    val rect = RectF(x, y, x + w, y + h)
                    detections.add(Detection(rect, label, confidence))
                 } else {
                    Log.w(TAG, "Skipping detection with invalid coordinates: x=$x, y=$y, w=$w, h=$h")
                 }
            }
        }
        if (detections.isNotEmpty()) {
            Log.i(TAG, "Detected ${detections.size} objects (after thresholding).")
        } else if (count > 0) {
             // Log if native reported count > 0 but Kotlin list is empty (due to thresholding or invalid data)
             Log.i(TAG, "Detected 0 objects after Kotlin thresholding/validation (native count was $count).")
        }
        return detections
    }

    // Check if Vulkan is supported and enabled
    fun isVulkanSupported(): Boolean {
        // Return the status determined during init
        return isInitialized && isVulkanAvailable
    }

    // Release NCNN resources. Call this when the detector is no longer needed (e.g., in onDestroy).
    fun release() {
        if (isInitialized) {
            try {
                releaseNative()
                isInitialized = false
                isModelLoaded = false
                isVulkanAvailable = false
                Log.i(TAG, "NCNN resources released successfully.")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Native release method not found", e)
            }
        }
    }
}
