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
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    // Lazy initialization of the detector
    private val detector: NcnnDetector by lazy {
        Log.d(TAG, "Initializing NcnnDetector instance...")
        NcnnDetector(this)
    }
    private lateinit var cameraExecutor: ExecutorService
    private val TAG = "MainActivity"

    // State for initialization status, using mutableStateOf for Compose reactivity
    private val isDetectorReady = mutableStateOf(false)
    private val initializationAttempted = AtomicBoolean(false)
    private var isVulkanUsed = false // Store whether Vulkan is active

    // Permission handling
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                Log.i(TAG, "Camera permission granted")
                // Re-trigger composition to potentially start camera setup
                setContent { AppContent() }
                // Attempt initialization again if permission was just granted and it failed before
                if (!isDetectorReady.value) {
                    initializeDetector()
                }
            } else {
                Log.e(TAG, "Camera permission denied")
                // Handle permission denial (e.g., show a message)
                // You might want to display a permanent message in the UI here
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize detector only if permission is already granted
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            initializeDetector()
        } else {
            Log.i(TAG, "Camera permission not yet granted. Requesting...")
            // Permission will be requested by AppContent
        }

        setContent { AppContent() }
    }

    private fun initializeDetector() {
        // Prevent multiple initialization attempts
        if (initializationAttempted.getAndSet(true)) {
            Log.d(TAG, "Initialization already attempted.")
            return
        }
        if (isDetectorReady.value) {
             Log.d(TAG, "Detector already initialized.")
             return
        }

        Log.i(TAG, "Starting detector initialization...")
        lifecycleScope.launch(Dispatchers.IO) { // Use IO dispatcher for potentially blocking native calls
            val initOk = detector.init()
            if (initOk) {
                isVulkanUsed = detector.isVulkanSupported() // Check Vulkan status after init
                Log.i(TAG, "Detector init successful. Loading model... (Vulkan: $isVulkanUsed)")
                val modelOk = detector.loadModel()
                if (modelOk) {
                    Log.i(TAG, "Model load successful.")
                    // Update state on the main thread for Compose
                    withContext(Dispatchers.Main) {
                        isDetectorReady.value = true
                        Log.i(TAG, "Detector is ready. isDetectorReady state is now: ${isDetectorReady.value}")
                    }
                } else {
                    Log.e(TAG, "Failed to load model.")
                    initializationAttempted.set(false) // Allow retry if model load fails
                }
            } else {
                Log.e(TAG, "Failed to initialize detector.")
                initializationAttempted.set(false) // Allow retry if init fails
            }
        }
    }

    @Composable
    private fun AppContent() {
        Yolo_kotlin_ncnnTheme {
            Surface(modifier = Modifier.fillMaxSize()) {
                val cameraPermissionGranted = ContextCompat.checkSelfPermission(
                    LocalContext.current, // Use LocalContext within Composable
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED

                if (cameraPermissionGranted) {
                    // Ensure initialization is triggered if needed (e.g., after permission grant)
                    LaunchedEffect(Unit) {
                        if (!isDetectorReady.value && !initializationAttempted.get()) {
                             initializeDetector()
                        }
                    }
                    // Pass the state value to CameraScreen
                    CameraScreen(
                        detector = detector,
                        cameraExecutor = cameraExecutor,
                        isDetectorReady = isDetectorReady.value, // Pass the current boolean value
                        isVulkanUsed = isVulkanUsed // Pass Vulkan status
                    )
                } else {
                    // Request permission if not granted
                    LaunchedEffect(Unit) {
                        Log.d(TAG, "Requesting camera permission from LaunchedEffect...")
                        requestPermissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                    PermissionRationale() // Show rationale while waiting/requesting
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "onDestroy called, releasing NCNN detector and shutting down camera executor.")
        // Release detector resources
        // Run release on a background thread if it might block
        // lifecycleScope.launch(Dispatchers.IO) { detector.release() } // Option 1: Coroutine
        cameraExecutor.execute { detector.release() } // Option 2: Use existing executor if appropriate

        // Shutdown executor
        if (!cameraExecutor.isShutdown) {
            cameraExecutor.shutdown()
            Log.d(TAG,"Camera executor shut down.")
        }
    }
}

@Composable
fun PermissionRationale() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
             Text("Camera permission is required.")
             Spacer(modifier = Modifier.height(8.dp))
             Text("Please grant permission to use the camera features.")
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraScreen(
    detector: NcnnDetector,
    cameraExecutor: ExecutorService,
    isDetectorReady: Boolean, // Receive the state value
    isVulkanUsed: Boolean     // Receive Vulkan status
) {
    // Log the value of isDetectorReady received by this composable instance
    Log.d("CameraScreen", "Composable recomposed/launched. isDetectorReady = $isDetectorReady, isVulkanUsed = $isVulkanUsed")

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    var detections by remember { mutableStateOf<List<NcnnDetector.Detection>>(emptyList()) }
    // Store the size of the image *sent* for detection (from ImageProxy)
    var sourceSize by remember { mutableStateOf(Size(0, 0)) }

    // FPS calculation state
    var fps by remember { mutableStateOf(0f) }
    val frameCount = remember { AtomicLong(0) }
    val lastProcessedFrameTimestamp = remember { AtomicLong(System.currentTimeMillis()) } // Track last processed frame time
    var lastFpsUpdateTime by remember { mutableStateOf(System.currentTimeMillis()) }
    val frameProcessingTimes = remember { mutableStateListOf<Long>() } // Track recent processing times
    val processingTimeAvgMs = remember { mutableStateOf(0f) }

    val previewView = remember { PreviewView(context).apply {
        scaleType = PreviewView.ScaleType.FIT_CENTER // Ensure consistent scaling
    }}
    val preview = remember { Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) } }

    // Make imageAnalyzer creation/update depend on isDetectorReady
    val imageAnalyzer = remember(isDetectorReady) { // Keyed on isDetectorReady
        Log.d("CameraScreen", "Creating/Updating ImageAnalysis instance (isDetectorReady=$isDetectorReady)")
        ImageAnalysis.Builder()
            // Set a resolution appropriate for the model input and performance.
            // 640x480 is a common choice. Higher resolution increases processing time.
            .setTargetResolution(Size(640, 480)) // Match native input aspect ratio if possible
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888) // Ensure YUV format for ImageUtils
            .build()
            .apply {
                if (isDetectorReady) {
                    Log.i("CameraScreen", "Detector is ready, setting analyzer.")
                    setAnalyzer(cameraExecutor) { imageProxy ->
                        val frameStartTime = System.currentTimeMillis()
                        // --- Frame Throttling (Optional Refinement) ---
                        // Consider dynamic throttling based on average processing time vs target FPS
                        // val targetFrameTimeMs = 33 // ~30 FPS target
                        // if (frameStartTime - lastProcessedFrameTimestamp.get() < targetFrameTimeMs) {
                        //     imageProxy.close()
                        //     return@setAnalyzer
                        // }
                        // --- Basic Throttling ---
                        // Skip frame if analysis is too fast (e.g., < 20ms) to prevent queue buildup
                        // Adjust based on observed performance. Lower value allows higher potential FPS.
                        val minFrameIntervalMs = 16 // Corresponds to ~60 FPS max attempt rate
                        if (frameStartTime - lastProcessedFrameTimestamp.get() < minFrameIntervalMs) {
                            imageProxy.close()
                            return@setAnalyzer
                        }
                        // --- End Throttling ---

                        val rotationDegrees = imageProxy.imageInfo.rotationDegrees // Needed if rotation is handled manually

                        // --- Image Conversion ---
                        val conversionStartTime = System.nanoTime()
                        // IMPORTANT: ImageUtils converts YUV_420_888 to RGBA ByteArray
                        val rgbaBytes = ImageUtils.imageProxyToRgbaByteArray(imageProxy)
                        val conversionEndTime = System.nanoTime()
                        val conversionTimeMs = (conversionEndTime - conversionStartTime) / 1_000_000

                        // Get dimensions *before* closing ImageProxy
                        val analysisWidth = imageProxy.width
                        val analysisHeight = imageProxy.height

                        // CRUCIAL: Close the ImageProxy ASAP after getting data
                        imageProxy.close()

                        if (rgbaBytes != null) {
                            // Update sourceSize only if it changes, reducing recompositions
                            if (sourceSize.width != analysisWidth || sourceSize.height != analysisHeight) {
                                sourceSize = Size(analysisWidth, analysisHeight)
                                Log.d("CameraScreen", "Source size updated: ${sourceSize.width}x${sourceSize.height}")
                            }

                            // --- Perform Detection ---
                            val detectStartTime = System.nanoTime()
                            // Pass the RGBA byte array and original dimensions
                            val results = detector.detect(rgbaBytes, analysisWidth, analysisHeight)
                            val detectEndTime = System.nanoTime()
                            val detectTimeMs = (detectEndTime - detectStartTime) / 1_000_000

                            // Update detections state (triggers recomposition for overlay)
                            detections = results

                            // --- Performance Logging & State Update ---
                            val frameEndTime = System.currentTimeMillis()
                            val totalFrameProcessingTime = frameEndTime - frameStartTime
                            lastProcessedFrameTimestamp.set(frameEndTime) // Use end time for next throttle check

                            // Update processing time average (e.g., over last 10 frames)
                            frameProcessingTimes.add(totalFrameProcessingTime)
                            if (frameProcessingTimes.size > 10) {
                                frameProcessingTimes.removeAt(0)
                            }
                            processingTimeAvgMs.value = frameProcessingTimes.average().toFloat()

                            // FPS calculation
                            val currentFrameCount = frameCount.incrementAndGet()
                            val elapsedFpsTime = frameEndTime - lastFpsUpdateTime
                            if (elapsedFpsTime >= 1000) { // Update FPS display every second
                                fps = (currentFrameCount * 1000f) / elapsedFpsTime
                                frameCount.set(0) // Reset frame count for the next second
                                lastFpsUpdateTime = frameEndTime
                                Log.d("CameraScreen", "Perf Update: FPS=%.1f | Avg Proc Time=%.1f ms (Convert= %d ms, Detect= %d ms)".format(
                                    fps, processingTimeAvgMs.value, conversionTimeMs, detectTimeMs))
                            }

                        } else {
                            Log.e("CameraScreen", "Could not convert ImageProxy to RGBA ByteArray. Format: ${imageProxy.format}")
                            // Close proxy here if not closed earlier in case of conversion failure
                            // imageProxy.close() // Already closed above
                        }
                    }
                } else {
                    // If detector is not ready, ensure no analyzer is set
                    Log.w("CameraScreen", "Detector not ready, clearing analyzer.")
                    clearAnalyzer()
                }
            }
    }

    // LaunchedEffect for binding camera use cases
    LaunchedEffect(cameraProviderFuture, preview, imageAnalyzer, isDetectorReady) { // Re-bind if analyzer or isDetectorReady changes
        Log.d("CameraScreen", "LaunchedEffect for binding camera use cases triggered (isDetectorReady=$isDetectorReady).")
        val cameraProvider = cameraProviderFuture.get()
        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            // Unbind existing use cases before rebinding
            cameraProvider.unbindAll()
            Log.d("CameraScreen", "Unbound all use cases.")

            // Bind the desired use cases
            // Only bind imageAnalyzer if it has an analyzer set (i.e., detector is ready)
            if (isDetectorReady) {
                 Log.d("CameraScreen", "Binding Preview and ImageAnalysis.")
                 cameraProvider.bindToLifecycle(
                     lifecycleOwner, cameraSelector, preview, imageAnalyzer
                 )
            } else {
                 Log.d("CameraScreen", "Binding Preview only (detector not ready).")
                 cameraProvider.bindToLifecycle(
                     lifecycleOwner, cameraSelector, preview
                 )
            }
            Log.i("CameraScreen", "Camera use cases bound successfully.")

        } catch (exc: Exception) {
            Log.e("CameraScreen", "Use case binding failed", exc)
        }
        // Return Unit explicitly or implicitly
    }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("YOLOv11 NCNN (${if(isVulkanUsed) "GPU" else "CPU"})") })
        }
    ) { paddingValues ->
        Box(modifier = Modifier.fillMaxSize().padding(paddingValues)) {
            AndroidView({ previewView }, modifier = Modifier.fillMaxSize())

            if (isDetectorReady) {
                // Draw bounding boxes only if the detector is ready
                BoundingBoxOverlay(
                    detections = detections,
                    sourceImageWidth = sourceSize.width, // Pass the original image width
                    sourceImageHeight = sourceSize.height, // Pass the original image height
                    modifier = Modifier.fillMaxSize()
                )
                // Display FPS and Avg Processing Time on top right
                Column(modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(8.dp)
                    .background(Color.Black.copy(alpha = 0.6f), MaterialTheme.shapes.small)
                    .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = "FPS: ${"%.1f".format(fps)}",
                        color = Color.White,
                        fontSize = 14.sp
                    )
                    Text(
                        text = "Proc: ${processingTimeAvgMs.value.roundToInt()} ms",
                        color = Color.White,
                        fontSize = 14.sp
                    )
                }

            } else {
                // Show loading indicator or message while detector is initializing
                Column(
                    modifier = Modifier.align(Alignment.Center),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.height(8.dp))
                    Text("Initializing Detector...")
                }
            }
        }
    }
}