package com.example.yolo_kotlin_ncnn

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
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
import androidx.compose.ui.unit.dp
import com.example.yolo_kotlin_ncnn.ui.theme.Yolo_kotlin_ncnnTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.util.Log

class MainActivity : ComponentActivity() {
    private val detector: NcnnDetector by lazy { NcnnDetector(this) }
    private val TAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

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
        Log.d(TAG, "onDestroy called, releasing NCNN detector.")
        detector.release()
    }
}

@Composable
fun NcnnStatusScreen(modifier: Modifier = Modifier, detector: NcnnDetector) {
    var isInitialized by remember { mutableStateOf<Boolean?>(null) }
    var isModelLoaded by remember { mutableStateOf<Boolean?>(null) }
    var hasVulkan by remember { mutableStateOf<Boolean?>(null) }
    var statusMessage by remember { mutableStateOf("Initializing...") }

    LaunchedEffect(key1 = detector) {
        statusMessage = "Initializing NCNN..."
        val initResult = withContext(Dispatchers.IO) {
            detector.init()
        }
        isInitialized = initResult

        if (initResult) {
            hasVulkan = detector.isVulkanSupported()
            statusMessage = "Loading model..."
            val loadResult = withContext(Dispatchers.IO) {
                detector.loadModel()
            }
            isModelLoaded = loadResult
            statusMessage = if (loadResult) "Initialization and model load complete." else "Model loading failed."
        } else {
            statusMessage = "NCNN Initialization failed."
            isModelLoaded = false
            hasVulkan = false
        }
    }

    Column(modifier = modifier.padding(16.dp)) {
        VulkanStatusChecker(
            isInitialized = isInitialized ?: false,
            isModelLoaded = isModelLoaded ?: false,
            hasVulkan = hasVulkan ?: false
        )

        Text(text = statusMessage, modifier = Modifier.padding(top = 8.dp))
    }
}

@Preview(showBackground = true)
@Composable
fun StatusPreview() {
    Yolo_kotlin_ncnnTheme {
        VulkanStatusChecker(
            isInitialized = false,
            isModelLoaded = false,
            hasVulkan = false
        )
    }
}