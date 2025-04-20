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
                Log.i(TAG, "Detector init successful. Loading model...")
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
                    // Optionally reset attempted flag if you want retry logic
                    // initializationAttempted.set(false)
                }
            } else {
                Log.e(TAG, "Failed to initialize detector.")
                // Optionally reset attempted flag if you want retry logic
                // initializationAttempted.set(false)
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
                        isDetectorReady = isDetectorReady.value // Pass the current boolean value
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
        detector.release()
        // Shutdown executor
        if (!cameraExecutor.isShutdown) {
            cameraExecutor.shutdown()
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
    isDetectorReady: Boolean // Receive the state value
) {
    // Log the value of isDetectorReady received by this composable instance
    Log.d("CameraScreen", "Composable recomposed/launched. isDetectorReady = $isDetectorReady")

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    var detections by remember { mutableStateOf<List<NcnnDetector.Detection>>(emptyList()) }
    // Store the size of the image *sent* for detection (from ImageProxy)
    var sourceSize by remember { mutableStateOf(Size(0, 0)) }

    // FPS calculation state
    var fps by remember { mutableStateOf(0f) }
    val frameCount = remember { AtomicLong(0) }
    val lastProcessedFrameTimestamp = remember { AtomicLong(0) } // Track last processed frame time
    var lastFpsUpdateTime by remember { mutableStateOf(System.currentTimeMillis()) }


    val previewView = remember { PreviewView(context) }
    val preview = remember { Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) } }

    // Make imageAnalyzer creation/update depend on isDetectorReady
    val imageAnalyzer = remember(isDetectorReady) { // Keyed on isDetectorReady
        Log.d("CameraScreen", "Creating/Updating ImageAnalysis instance (isDetectorReady=$isDetectorReady)")
        ImageAnalysis.Builder()
            // Set a resolution appropriate for the model input and performance.
            // Using a fixed size might be better than relying on device defaults.
            .setTargetResolution(Size(640, 480)) // Example: Common camera resolution
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888) // Ensure YUV format for ImageUtils
            .build()
            .apply {
                if (isDetectorReady) {
                    Log.i("CameraScreen", "Detector is ready, setting analyzer.")
                    setAnalyzer(cameraExecutor) { imageProxy ->
                        val currentTime = System.currentTimeMillis()
                        // Basic throttling: Skip frame if analysis is too fast (e.g., < 30ms)
                        // to prevent overwhelming the detector and allow UI updates. Adjust as needed.
                        if (currentTime - lastProcessedFrameTimestamp.get() < 30) { // Approx 33 FPS limit
                            imageProxy.close()
                            return@setAnalyzer
                        }
                        lastProcessedFrameTimestamp.set(currentTime)


                        // FPS calculation
                        val currentFrameCount = frameCount.incrementAndGet()
                        val elapsedFpsTime = currentTime - lastFpsUpdateTime
                        if (elapsedFpsTime >= 1000) { // Update FPS display every second
                            fps = (currentFrameCount * 1000f) / elapsedFpsTime
                            frameCount.set(0) // Reset frame count for the next second
                            lastFpsUpdateTime = currentTime
                            Log.d("CameraScreen", "FPS Updated: ${"%.1f".format(fps)}")
                        }

                        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                        // Convert ImageProxy (YUV_420_888) to RGBA ByteArray
                        val conversionStartTime = System.nanoTime()
                        val rgbaBytes = ImageUtils.imageProxyToRgbaByteArray(imageProxy)
                        val conversionEndTime = System.nanoTime()
                        // Log conversion time occasionally
                        // if (frameCount.get() % 30 == 0L) { // Log every 30 frames approx
                        //     Log.d("CameraScreen", "YUV->RGBA conversion time: ${(conversionEndTime - conversionStartTime) / 1_000_000} ms")
                        // }


                        if (rgbaBytes != null) {
                            // Get the dimensions of the image *before* potential resizing in native code
                            val analysisWidth = imageProxy.width
                            val analysisHeight = imageProxy.height
                            // Update sourceSize only if it changes, reducing recompositions
                            if (sourceSize.width != analysisWidth || sourceSize.height != analysisHeight) {
                                sourceSize = Size(analysisWidth, analysisHeight)
                                Log.d("CameraScreen", "Source size updated: ${sourceSize.width}x${sourceSize.height}")
                            }

                            // Perform detection using the RGBA byte array and original dimensions
                            val results = detector.detect(rgbaBytes, analysisWidth, analysisHeight)
                            detections = results // Update detections state

                        } else {
                            Log.e("CameraScreen", "Could not convert ImageProxy to RGBA ByteArray. Format: ${imageProxy.format}")
                        }

                        imageProxy.close() // CRUCIAL: Close the ImageProxy
                    }
                } else {
                    // If detector is not ready, ensure no analyzer is set
                    Log.w("CameraScreen", "Detector not ready, clearing analyzer.")
                    clearAnalyzer()
                }
            }
    }

    // LaunchedEffect for binding camera use cases
    LaunchedEffect(cameraProviderFuture, preview, imageAnalyzer) { // Re-bind if analyzer instance changes
        Log.d("CameraScreen", "LaunchedEffect for binding camera use cases triggered.")
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
            TopAppBar(title = { Text("YOLOv11 NCNN Demo") })
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
                // Display FPS on top right
                Text(
                    text = "FPS: ${"%.1f".format(fps)}",
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(8.dp)
                        .background(Color.Black.copy(alpha = 0.6f), MaterialTheme.shapes.small)
                        .padding(horizontal = 8.dp, vertical = 4.dp),
                    color = Color.White,
                    fontSize = 16.sp
                )
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