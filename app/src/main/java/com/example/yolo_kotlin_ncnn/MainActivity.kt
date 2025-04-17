package com.example.yolo_kotlin_ncnn

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.yolo_kotlin_ncnn.ui.theme.Yolo_kotlin_ncnnTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

class MainActivity : ComponentActivity() {
    private val detector: NcnnDetector by lazy { NcnnDetector(this) }
    private lateinit var cameraExecutor: ExecutorService
    private val TAG = "MainActivity"

    // State for initialization status
    private val isDetectorReady = mutableStateOf(false)

    // Permission handling
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                Log.i(TAG, "Camera permission granted")
                // Recompose to trigger camera setup if permission was just granted
                setContent { AppContent() }
            } else {
                Log.e(TAG, "Camera permission denied")
                // Handle permission denial (e.g., show a message)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize detector asynchronously
        lifecycleScope.launch(Dispatchers.IO) {
            Log.i(TAG, "Starting detector initialization...") // Log start
            val initOk = detector.init()
            if (initOk) {
                Log.i(TAG, "Detector init successful. Loading model...") // Log init success
                val modelOk = detector.loadModel()
                if (modelOk) {
                    // *** ADDED LOGGING ***
                    Log.i(TAG, "Model load successful. Setting isDetectorReady = true")
                    // *** END ADDED LOGGING ***
                    isDetectorReady.value = true
                    // *** ADDED LOGGING ***
                    Log.i(TAG, "isDetectorReady state is now: ${isDetectorReady.value}")
                    // *** END ADDED LOGGING ***
                    Log.i(TAG, "Detector initialized and model loaded successfully.") // Keep original log
                } else {
                    Log.e(TAG, "Failed to load model.")
                }
            } else {
                Log.e(TAG, "Failed to initialize detector.")
            }
        }

        setContent { AppContent() }
    }

    @Composable
    private fun AppContent() {
        Yolo_kotlin_ncnnTheme {
            Surface(modifier = Modifier.fillMaxSize()) {
                when (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)) {
                    PackageManager.PERMISSION_GRANTED -> {
                        CameraScreen(detector = detector, cameraExecutor = cameraExecutor, isDetectorReady = isDetectorReady.value)
                    }
                    else -> {
                        LaunchedEffect(Unit) {
                            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
                        }
                        PermissionRationale() // Show rationale while waiting for permission result
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "onDestroy called, releasing NCNN detector and shutting down camera executor.")
        detector.release()
        cameraExecutor.shutdown()
    }
}

@Composable
fun PermissionRationale() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("Camera permission is required to use this feature.")
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraScreen(
    detector: NcnnDetector,
    cameraExecutor: ExecutorService,
    isDetectorReady: Boolean
) {
    // *** ADDED LOGGING ***
    // Log the value of isDetectorReady received by this composable instance
    Log.d("CameraScreen", "Composable recomposed/launched. isDetectorReady = $isDetectorReady")
    // *** END ADDED LOGGING ***

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    var detections by remember { mutableStateOf<List<NcnnDetector.Detection>>(emptyList()) }
    var sourceSize by remember { mutableStateOf(Size(0, 0)) } // Store the size of the image sent for detection

    // FPS calculation state
    var fps by remember { mutableStateOf(0f) }
    val frameCount = remember { AtomicLong(0) }
    var lastFpsTimestamp by remember { mutableStateOf(System.currentTimeMillis()) }

    val previewView = remember { PreviewView(context) }
    val preview = remember { Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) } }

    // *** Make imageAnalyzer depend on isDetectorReady ***
    val imageAnalyzer = remember(isDetectorReady) { // Add isDetectorReady as a key
        Log.d("CameraScreen", "Creating/Recreating ImageAnalysis (isDetectorReady=$isDetectorReady)") // Log recreation
        ImageAnalysis.Builder()
            .setTargetResolution(Size(640, 640)) // Match model input if possible, adjust as needed
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .apply {
                // Set the analyzer only if the detector is ready when this block executes
                if (isDetectorReady) {
                    Log.d("CameraScreen", "Setting Analyzer because isDetectorReady is true")
                    setAnalyzer(cameraExecutor) { imageProxy ->
                        // FPS calculation
                        val currentFrameCount = frameCount.incrementAndGet()
                        val currentTime = System.currentTimeMillis()
                        val elapsedTime = currentTime - lastFpsTimestamp
                        if (elapsedTime >= 1000) { // Update FPS every second
                            fps = (currentFrameCount * 1000f) / elapsedTime
                            frameCount.set(0) // Reset frame count
                            lastFpsTimestamp = currentTime
                            // *** ADDED LOGGING ***
                            Log.d("CameraScreen", "FPS Updated: ${"%.1f".format(fps)}")
                            // *** END ADDED LOGGING ***
                        }

                        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                        // Convert ImageProxy to RGBA ByteArray
                        val rgbaBytes = ImageUtils.imageProxyToRgbaByteArray(imageProxy)

                        if (rgbaBytes != null) {
                            // *** ADDED LOGGING ***
                            Log.d("CameraScreen", "RGBA ByteArray conversion successful. Size: ${rgbaBytes.size}, Rotation: $rotationDegrees")
                            // *** END ADDED LOGGING ***

                            // Store the size of the image being analyzed (original YUV dimensions)
                            val analysisWidth = imageProxy.width
                            val analysisHeight = imageProxy.height
                            sourceSize = Size(analysisWidth, analysisHeight)
                            // *** ADDED LOGGING ***
                            Log.d("CameraScreen", "Source size updated: ${sourceSize.width}x${sourceSize.height}. Calling detector.detect...")
                            // *** END ADDED LOGGING ***

                            // Perform detection using the RGBA byte array
                            // Pass the original width and height
                            val results = detector.detect(rgbaBytes, analysisWidth, analysisHeight)
                            // *** ADDED LOGGING ***
                            Log.d("CameraScreen", "Detector returned ${results.size} results.")
                            // *** END ADDED LOGGING ***
                            detections = results

                        } else {
                            Log.e("CameraScreen", "Could not convert ImageProxy to RGBA ByteArray. Image format: ${imageProxy.format}")
                        }

                        imageProxy.close() // Crucial to close the ImageProxy
                    }
                } else {
                    // If detector is not ready when analyzer is created, clear any previous analyzer
                    Log.d("CameraScreen", "Clearing Analyzer because isDetectorReady is false")
                    clearAnalyzer()
                }
            }
    }

    // LaunchedEffect for binding now depends on imageAnalyzer instance
    LaunchedEffect(cameraProviderFuture, imageAnalyzer) { // Add imageAnalyzer as a key
        Log.d("CameraScreen", "LaunchedEffect for binding triggered.")
        val cameraProvider = cameraProviderFuture.get()
        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            // Unbind use cases before rebinding
            cameraProvider.unbindAll()

            // Bind use cases to camera
            cameraProvider.bindToLifecycle(
                lifecycleOwner, cameraSelector, preview, imageAnalyzer
            )
        } catch (exc: Exception) {
            Log.e("CameraScreen", "Use case binding failed", exc)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("YOLOv11 NCNN Demo") })
        }
    ) { paddingValues ->
        Box(modifier = Modifier.fillMaxSize().padding(paddingValues)) {
            AndroidView({ previewView }, modifier = Modifier.fillMaxSize())
            if (isDetectorReady) {
                // *** ADDED LOGGING ***
                Log.d("CameraScreen", "Passing ${detections.size} detections to BoundingBoxOverlay. Source size: ${sourceSize.width}x${sourceSize.height}")
                // *** END ADDED LOGGING ***
                BoundingBoxOverlay(
                    detections = detections,
                    sourceImageWidth = sourceSize.width,
                    sourceImageHeight = sourceSize.height,
                    modifier = Modifier.fillMaxSize()
                )
                // Display FPS on top right
                Text(
                    text = "FPS: ${"%.1f".format(fps)}",
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(8.dp)
                        .background(Color.Black.copy(alpha = 0.5f), MaterialTheme.shapes.small)
                        .padding(horizontal = 8.dp, vertical = 4.dp),
                    color = Color.White,
                    fontSize = 16.sp
                )
            } else {
                // Show loading indicator while detector is initializing
                CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
            }
        }
    }
}