package com.example.yolo_kotlin_ncnn

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
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
    
    Column(modifier = modifier) {
        Text(
            text = "NCNN Status",
            style = MaterialTheme.typography.headlineMedium
        )
        Text("Initialization: ${if (isInitialized) "Success" else "Failed"}")
        Text("Model loaded: ${if (isModelLoaded) "Yes" else "No"}")
        Text("Vulkan support: ${if (hasVulkan) "Available" else "Unavailable"}")
    }
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    Yolo_kotlin_ncnnTheme {
        // Can't preview with actual detector
        Column {
            Text("NCNN Status")
            Text("Initialization: Success")
            Text("Model loaded: Yes")
            Text("Vulkan support: Available")
        }
    }
}