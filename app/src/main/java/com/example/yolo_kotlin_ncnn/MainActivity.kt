package com.example.yolo_kotlin_ncnn

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.ImageFormat
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
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.yolo_kotlin_ncnn.ui.theme.Yolo_kotlin_ncnnTheme
import kotlinx.coroutines.*
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.roundToInt

// Define states for initialization status
internal enum class InitStatus { // Changed from private to internal
    IDLE, PENDING, SUCCESS, FAILED
}

class MainActivity : ComponentActivity() {
    // Lazy initialization of the detector
    private val detector: NcnnDetector by lazy {
        Log.d(TAG, "Initializing NcnnDetector instance...")
        NcnnDetector(this)
    }
    private lateinit var cameraExecutor: ExecutorService
    private val TAG = "MainActivity"

    // State for initialization status, using mutableStateOf for Compose reactivity
    private val initializationStatus = mutableStateOf(InitStatus.IDLE)
    private var initErrorMessage = mutableStateOf<String?>(null) // Store error message

    // Permission handling
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                Log.i(TAG, "Camera permission granted")
                // Re-trigger composition to potentially start camera setup
                setContent { AppContent() } // Recompose to reflect permission change
                // Attempt initialization if not already successful or pending
                if (initializationStatus.value == InitStatus.IDLE || initializationStatus.value == InitStatus.FAILED) {
                    initializeDetector()
                }
            } else {
                Log.e(TAG, "Camera permission denied")
                // Update UI to show persistent denial message
                setContent {
                    Yolo_kotlin_ncnnTheme {
                        Surface(modifier = Modifier.fillMaxSize()) {
                            PermissionRationale(isDeniedPermanently = true)
                        }
                    }
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Check permission and initialize if granted, otherwise AppContent will request
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            initializeDetector()
        } else {
            Log.i(TAG, "Camera permission not yet granted. Requesting via UI.")
        }

        setContent { AppContent() }
    }

    private fun initializeDetector() {
        // Prevent multiple concurrent initializations
        if (initializationStatus.value == InitStatus.PENDING || initializationStatus.value == InitStatus.SUCCESS) {
            Log.d(TAG, "Initialization already pending or successful.")
            return
        }

        Log.i(TAG, "Starting detector initialization...")
        initializationStatus.value = InitStatus.PENDING
        initErrorMessage.value = null // Clear previous errors

        lifecycleScope.launch(Dispatchers.IO) { // Use IO dispatcher for native calls
            try {
                val initOk = detector.init()
                if (initOk) {
                    Log.i(TAG, "Detector init successful. Loading model... (Vulkan: ${detector.hasVulkanGpu})")
                    val modelOk = detector.loadModel()
                    if (modelOk) {
                        Log.i(TAG, "Model load successful.")
                        withContext(Dispatchers.Main) {
                            initializationStatus.value = InitStatus.SUCCESS
                            Log.i(TAG, "Detector is ready. Initialization status: ${initializationStatus.value}")
                        }
                    } else {
                        Log.e(TAG, "Failed to load model.")
                        withContext(Dispatchers.Main) {
                            initializationStatus.value = InitStatus.FAILED
                            initErrorMessage.value = "Failed to load model. Check logs and asset files."
                        }
                    }
                } else {
                    Log.e(TAG, "Failed to initialize detector.")
                    withContext(Dispatchers.Main) {
                        initializationStatus.value = InitStatus.FAILED
                        initErrorMessage.value = "Failed to initialize NCNN. Check logs."
                    }
                }
            } catch (e: Exception) {
                 Log.e(TAG, "Exception during detector initialization", e)
                 withContext(Dispatchers.Main) {
                     initializationStatus.value = InitStatus.FAILED
                     initErrorMessage.value = "Error during initialization: ${e.localizedMessage}"
                 }
            }
        }
    }

    @Composable
    private fun AppContent() {
        Yolo_kotlin_ncnnTheme {
            Surface(modifier = Modifier.fillMaxSize()) {
                val cameraPermissionGranted = ContextCompat.checkSelfPermission(
                    LocalContext.current,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED

                if (cameraPermissionGranted) {
                    // Ensure initialization is triggered if needed (e.g., after permission grant or config change)
                    LaunchedEffect(Unit) { // Runs once when entering composition if permission is granted
                        if (initializationStatus.value == InitStatus.IDLE) {
                             initializeDetector()
                        }
                    }
                    // Pass the state value to CameraScreen
                    CameraScreen(
                        detector = detector,
                        cameraExecutor = cameraExecutor,
                        initStatus = initializationStatus.value, // Pass the current status enum
                        isNcnnInitialized = detector.isInitialized,
                        isModelLoaded = detector.isModelLoaded,
                        isVulkanUsed = detector.hasVulkanGpu,
                        errorMessage = initErrorMessage.value // Pass error message
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
        // Release detector resources safely
        // Use a coroutine or the executor, ensure it completes if possible
        lifecycleScope.launch(Dispatchers.IO) {
             try {
                 detector.release() // release() has internal checks
             } catch (e: Exception) {
                 Log.e(TAG, "Exception during detector release", e)
             }
        }

        // Shutdown executor
        if (!cameraExecutor.isShutdown) {
            try {
                cameraExecutor.shutdown()
                Log.d(TAG,"Camera executor shut down.")
            } catch (e: SecurityException) {
                Log.e(TAG, "Failed to shutdown camera executor", e)
            }
        }
    }
}

@Composable
fun PermissionRationale(isDeniedPermanently: Boolean = false) {
    Box(modifier = Modifier.fillMaxSize().padding(16.dp), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
             Text("Camera Permission Required", style = MaterialTheme.typography.headlineSmall)
             Spacer(modifier = Modifier.height(16.dp))
             Text(
                 if (isDeniedPermanently) "Camera permission was denied. Please enable it in app settings to use this feature."
                 else "This app needs camera access to perform real-time object detection.",
                 textAlign = TextAlign.Center
             )
             // Optionally add a button to open app settings if permanently denied
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun CameraScreen( // Changed visibility to internal
    detector: NcnnDetector,
    cameraExecutor: ExecutorService,
    initStatus: InitStatus, // Receive the initialization status
    isNcnnInitialized: Boolean,
    isModelLoaded: Boolean,
    isVulkanUsed: Boolean,
    errorMessage: String?     // Receive potential error message
) {
    val isDetectorReady = initStatus == InitStatus.SUCCESS // Derived state

    // Log the value of isDetectorReady received by this composable instance
    Log.d("CameraScreen", "Composable recomposed/launched. InitStatus = $initStatus, isDetectorReady = $isDetectorReady, isVulkanUsed = $isVulkanUsed")

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope() // Get a CoroutineScope tied to the composable lifecycle
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    var detections by remember { mutableStateOf<List<Detection>>(emptyList()) } // Updated type
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
    val imageAnalyzer = remember(isDetectorReady, detector, scope) { // Keyed on isDetectorReady, detector, scope
        Log.d("CameraScreen", "Creating/Updating ImageAnalysis instance (isDetectorReady=$isDetectorReady)")
        ImageAnalysis.Builder()
            // Request a resolution suitable for the model, e.g., 640x480 or similar aspect ratio
            // Higher resolution increases processing time.
            .setTargetResolution(Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            // IMPORTANT: Request YUV_420_888 format for direct buffer access
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .apply {
                if (isDetectorReady) {
                    Log.i("CameraScreen", "Detector is ready, setting analyzer.")
                    setAnalyzer(cameraExecutor) { imageProxy ->
                        val frameStartTime = System.currentTimeMillis()

                        // --- Basic Throttling (Optional but recommended) ---
                        val minFrameIntervalMs = 16 // ~60 FPS max attempt rate
                        if (frameStartTime - lastProcessedFrameTimestamp.get() < minFrameIntervalMs) {
                            imageProxy.close() // Close the image quickly
                            return@setAnalyzer
                        }
                        // --- End Throttling ---

                        val rotationDegrees = imageProxy.imageInfo.rotationDegrees // Needed if model doesn't handle rotation
                        val analysisWidth = imageProxy.width
                        val analysisHeight = imageProxy.height

                        // --- Extract YUV Data ---
                        // Check if the format is indeed YUV_420_888
                        if (imageProxy.format == ImageFormat.YUV_420_888) {
                            val planes = imageProxy.planes
                            val yBuffer = planes[0].buffer
                            val uBuffer = planes[1].buffer // U plane
                            val vBuffer = planes[2].buffer // V plane

                            val yStride = planes[0].rowStride
                            val uvStride = planes[1].rowStride // U/V planes usually have same stride
                            val uvPixelStride = planes[1].pixelStride // U/V planes usually have same pixel stride

                            // Close the imageProxy *after* extracting buffers and info
                            imageProxy.close()

                            // Launch detection in a coroutine, passing YUV data
                            scope.launch { // Use the composable's scope
                                val detectStartTime = System.nanoTime()

                                // Call the public suspend function that takes YUV ByteBuffers
                                val results: List<Detection>? = detector.detect(
                                    yBuffer, uBuffer, vBuffer,
                                    yStride, uvStride, uvPixelStride,
                                    analysisWidth, analysisHeight
                                    // Pass rotationDegrees if needed by native code
                                )
                                val detectEndTime = System.nanoTime()
                                val detectTimeMs = (detectEndTime - detectStartTime) / 1_000_000

                                // Update state on the main thread
                                withContext(Dispatchers.Main) {
                                    if (sourceSize.width != analysisWidth || sourceSize.height != analysisHeight) {
                                        sourceSize = Size(analysisWidth, analysisHeight)
                                        Log.d("CameraScreen", "Source size updated: ${sourceSize.width}x${sourceSize.height}")
                                    }
                                    // Update detections state
                                    results?.let { detections = it }

                                    // --- Performance Logging & State Update ---
                                    val frameEndTime = System.currentTimeMillis()
                                    val totalFrameProcessingTime = frameEndTime - frameStartTime
                                    lastProcessedFrameTimestamp.set(frameEndTime)

                                    frameProcessingTimes.add(totalFrameProcessingTime)
                                    if (frameProcessingTimes.size > 10) frameProcessingTimes.removeAt(0)
                                    processingTimeAvgMs.value = frameProcessingTimes.average().toFloat()

                                    val currentFrameCount = frameCount.incrementAndGet()
                                    val elapsedFpsTime = frameEndTime - lastFpsUpdateTime
                                    if (elapsedFpsTime >= 1000) {
                                        fps = (currentFrameCount * 1000f) / elapsedFpsTime
                                        frameCount.set(0)
                                        lastFpsUpdateTime = frameEndTime
                                        // Log detect time separately now
                                        Log.d("CameraScreen", "Perf Update: FPS=%.1f | Avg Proc Time=%.1f ms (Detect=%d ms)".format(
                                            fps, processingTimeAvgMs.value, detectTimeMs))
                                    }
                                } // End withContext(Dispatchers.Main)
                            } // End scope.launch
                        } else {
                            // Format is not YUV_420_888, handle error or skip frame
                            Log.e("CameraScreen", "Unexpected image format: ${imageProxy.format}. Expected YUV_420_888.")
                            imageProxy.close() // Ensure image is closed
                            // Optionally clear detections
                            scope.launch(Dispatchers.Main) { detections = emptyList() }
                        }
                    } // End setAnalyzer lambda
                } else {
                    // If detector is not ready, ensure no analyzer is set
                    Log.w("CameraScreen", "Detector not ready, clearing analyzer.")
                    clearAnalyzer()
                }
            } // End apply
    } // End remember for imageAnalyzer

    // LaunchedEffect for binding camera use cases
    LaunchedEffect(cameraProviderFuture, preview, imageAnalyzer, initStatus) { // Re-bind if analyzer or initStatus changes
        Log.d("CameraScreen", "LaunchedEffect for binding camera use cases triggered (InitStatus=$initStatus).")
        val cameraProvider = cameraProviderFuture.get()
        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            cameraProvider.unbindAll()
            Log.d("CameraScreen", "Unbound all use cases.")

            // Bind use cases based on detector readiness
            val useCasesToBind = mutableListOf<UseCase>(preview)
            if (initStatus == InitStatus.SUCCESS) {
                useCasesToBind.add(imageAnalyzer)
                Log.d("CameraScreen", "Binding Preview and ImageAnalysis.")
            } else {
                Log.d("CameraScreen", "Binding Preview only (detector not ready or failed).")
            }

            cameraProvider.bindToLifecycle(
                 lifecycleOwner, cameraSelector, *useCasesToBind.toTypedArray()
            )
            Log.i("CameraScreen", "Camera use cases bound successfully.")

        } catch (exc: Exception) {
            Log.e("CameraScreen", "Use case binding failed", exc)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("YOLOv11 NCNN (${if(isVulkanUsed) "GPU" else "CPU"})") })
        }
    ) { paddingValues ->
        Box(modifier = Modifier.fillMaxSize().padding(paddingValues)) {
            AndroidView({ previewView }, modifier = Modifier.fillMaxSize())

            // Overlay content based on initialization status
            when (initStatus) {
                InitStatus.SUCCESS -> {
                    // Draw bounding boxes only if the detector is ready
                    BoundingBoxOverlay(
                        detections = detections,
                        sourceImageWidth = sourceSize.width,
                        sourceImageHeight = sourceSize.height,
                        modifier = Modifier.fillMaxSize()
                    )
                    // Display FPS and Avg Processing Time
                    Column(modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(8.dp)
                        .background(Color.Black.copy(alpha = 0.6f), MaterialTheme.shapes.small)
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                    ) {
                        Text("FPS: ${"%.1f".format(fps)}", color = Color.White, fontSize = 14.sp)
                        Text("Proc: ${processingTimeAvgMs.value.roundToInt()} ms", color = Color.White, fontSize = 14.sp)
                    }
                }
                InitStatus.PENDING -> {
                    // Show loading indicator
                    Column(
                        modifier = Modifier.align(Alignment.Center),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        CircularProgressIndicator()
                        Spacer(modifier = Modifier.height(8.dp))
                        Text("Initializing Detector...")
                    }
                }
                InitStatus.FAILED -> {
                    // Show error message
                     Column(
                        modifier = Modifier.align(Alignment.Center).padding(16.dp)
                            .background(Color.Black.copy(alpha = 0.7f), MaterialTheme.shapes.medium)
                            .padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text("Initialization Failed", color = Color.Red, style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(errorMessage ?: "An unknown error occurred.", color = Color.White, textAlign = TextAlign.Center)
                        // Optionally add a retry button
                        // Button(onClick = { /* Call initializeDetector() again */ }) { Text("Retry") }
                    }
                }
                InitStatus.IDLE -> {
                     // Show placeholder or message if needed (e.g., waiting for permission)
                     // This state might not be visible for long if permission is granted quickly
                     Text("Waiting for initialization...", modifier = Modifier.align(Alignment.Center))
                }
            }
        }
    }
}