package com.example.yolo_kotlin_ncnn

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import java.nio.ByteBuffer
import android.util.Log

class NcnnDetector(private val context: Context) {
    
    data class Detection(
        val rect: RectF,
        val label: Int,
        val confidence: Float
    )
    
    companion object {
        private const val TAG = "NcnnDetector"
        
        // Load the native library
        init {
            System.loadLibrary("native-lib")
        }
    }
    
    // Native methods
    external fun initNative(assetManager: AssetManager): Boolean
    external fun loadModel(assetManager: AssetManager): Boolean
    external fun hasVulkan(): Boolean
    external fun detect(imageBytes: ByteArray, width: Int, height: Int): FloatArray
    external fun releaseNative()
    
    private var isInitialized = false
    private var isModelLoaded = false
    
    // Initialize detector
    fun init(): Boolean {
        if (isInitialized) return true
        
        isInitialized = initNative(context.assets)
        if (isInitialized) {
            Log.i(TAG, "NCNN initialized successfully")
            Log.i(TAG, "Vulkan support: ${if (hasVulkan()) "available" else "unavailable"}")
        } else {
            Log.e(TAG, "NCNN initialization failed")
        }
        
        return isInitialized
    }
    
    // Load YOLO model
    fun loadModel(): Boolean {
        if (!isInitialized) {
            if (!init()) return false
        }
        
        isModelLoaded = loadModel(context.assets)
        if (isModelLoaded) {
            Log.i(TAG, "YOLO model loaded successfully")
        } else {
            Log.e(TAG, "Failed to load YOLO model")
        }
        
        return isModelLoaded
    }
    
    // Detect objects in bitmap
    fun detect(bitmap: Bitmap, threshold: Float = 0.5f): List<Detection> {
        if (!isModelLoaded) {
            if (!loadModel()) return emptyList()
        }
        
        // Convert bitmap to RGBA bytes
        val bytes = ByteArray(bitmap.width * bitmap.height * 4)
        val buffer = ByteBuffer.wrap(bytes)
        bitmap.copyPixelsToBuffer(buffer)
        
        // Run detection
        val result = detect(bytes, bitmap.width, bitmap.height)
        
        // Parse results format: [count, x, y, width, height, label, confidence, ...]
        val detections = mutableListOf<Detection>()
        val count = result[0].toInt()
        
        for (i in 0 until count) {
            val offset = 1 + i * 6
            val x = result[offset]
            val y = result[offset + 1]
            val width = result[offset + 2]
            val height = result[offset + 3]
            val label = result[offset + 4].toInt()
            val confidence = result[offset + 5]
            
            if (confidence > threshold) {
                val rect = RectF(
                    (x - width/2) * bitmap.width,
                    (y - height/2) * bitmap.height,
                    (x + width/2) * bitmap.width,
                    (y + height/2) * bitmap.height
                )
                
                detections.add(Detection(rect, label, confidence))
            }
        }
        
        return detections
    }
    
    // Check if Vulkan is supported
    fun isVulkanSupported(): Boolean {
        return if (isInitialized) hasVulkan() else false
    }
    
    // Release resources
    fun release() {
        if (isInitialized) {
            releaseNative()
            isInitialized = false
            isModelLoaded = false
            Log.i(TAG, "NCNN resources released")
        }
    }
}
