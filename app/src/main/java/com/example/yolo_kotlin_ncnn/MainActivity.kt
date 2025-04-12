package com.example.yolo_kotlin_ncnn

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.yolo_kotlin_ncnn.ui.theme.Yolo_kotlin_ncnnTheme

class MainActivity : ComponentActivity() {
    private lateinit var detector: NcnnDetector
    private val TAG = "MainActivity"
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        
        // Initialize detector
        detector = NcnnDetector(this)
        
        setContent {
            Yolo_kotlin_ncnnTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    // Use the improved Vulkan status checker UI
                    NcnnStatusScreen(
                        modifier = Modifier.padding(innerPadding),
                        detector = detector
                    )
                }
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        detector.release()
    }
}

@Composable
fun NcnnStatusScreen(modifier: Modifier = Modifier, detector: NcnnDetector) {
    var isInitialized by remember { mutableStateOf(false) }
    var isModelLoaded by remember { mutableStateOf(false) }
    var hasVulkan by remember { mutableStateOf(false) }
    
    LaunchedEffect(key1 = detector) {
        isInitialized = detector.init()
        if (isInitialized) {
            isModelLoaded = detector.loadModel()
            hasVulkan = detector.isVulkanSupported()
        }
    }
    
    Column(modifier = modifier.padding(16.dp)) {
        // Use the VulkanStatusChecker imported from its own file
        VulkanStatusChecker(
            isInitialized = isInitialized,
            isModelLoaded = isModelLoaded,
            hasVulkan = hasVulkan
        )
    }
}

@Preview(showBackground = true)
@Composable
fun StatusPreview() {
    Yolo_kotlin_ncnnTheme {
        // This now refers to the VulkanStatusChecker from VulkanStatusChecker.kt
        VulkanStatusChecker(
            isInitialized = true,
            isModelLoaded = true,
            hasVulkan = true
        )
    }
}