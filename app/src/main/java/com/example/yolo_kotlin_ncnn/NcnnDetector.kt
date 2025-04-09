package com.example.yolo_kotlin_ncnn

import android.content.Context
import android.content.res.AssetManager
import java.io.File

class NcnnDetector(private val context: Context) {
    
    companion object {
        // Load the native library
        init {
            System.loadLibrary("native-lib")
        }
    }
    
    // Native methods
    external fun initNative(assetManager: AssetManager): Boolean
    external fun releaseNative()
    
    // Initialize detector
    fun init(): Boolean {
        return initNative(context.assets)
    }
    
    // Release resources
    fun release() {
        releaseNative()
    }
}
