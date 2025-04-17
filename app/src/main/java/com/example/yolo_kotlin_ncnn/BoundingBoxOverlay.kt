package com.example.yolo_kotlin_ncnn

import android.util.Log // Ensure import exists
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

private const val TAG = "BoundingBoxOverlay" // Logging tag

// Simple color cache for classes
private val classColors = mutableMapOf<Int, Color>()
private fun getColorForClass(classId: Int): Color {
    return classColors.getOrPut(classId) {
        // Generate visually distinct colors if possible
        val hue = (classId * 60) % 360 // Spread hues
        Color.hsv(hue = hue.toFloat(), saturation = 0.8f, value = 0.9f, alpha = 1.0f)
    }
}

// TODO: Replace with your actual class names if available
private val cocoClassNames = listOf(
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "pump", "pipe",
    "steel pipe", "electric cable"
)

@Composable
fun BoundingBoxOverlay(
    detections: List<NcnnDetector.Detection>,
    sourceImageWidth: Int,
    sourceImageHeight: Int,
    modifier: Modifier = Modifier
) {
    val density = LocalDensity.current
    val strokeWidthPx = with(density) { 2.dp.toPx() } // Stroke width for boxes
    val textMeasurer = rememberTextMeasurer()

    Canvas(modifier = modifier.fillMaxSize()) {
        // *** ADDED TEST DRAW ***
        // Draw a fixed red square in the center to test if Canvas is rendering
        val canvasCenterX = size.width / 2f
        val canvasCenterY = size.height / 2f
        drawRect(
            color = Color.Red,
            topLeft = Offset(canvasCenterX - 50f, canvasCenterY - 50f),
            size = Size(100f, 100f),
            style = Stroke(width = 5f)
        )
        // Log that the test draw is attempted
        Log.d(TAG, "Attempting to draw test rectangle.")
        // *** END ADDED TEST DRAW ***

        if (sourceImageWidth <= 0 || sourceImageHeight <= 0) {
            Log.w(TAG, "Skipping draw: Invalid source image dimensions ($sourceImageWidth x $sourceImageHeight)")
            return@Canvas
        }

        val canvasWidth = size.width
        val canvasHeight = size.height

        // --- Scaling Logic ---
        // This logic assumes the camera preview (like PreviewView) uses a scale type
        // equivalent to FIT_CENTER. It scales the source image dimensions to fit within
        // the canvas while maintaining aspect ratio, then centers the result.
        val scaleX = canvasWidth / sourceImageWidth.toFloat()
        val scaleY = canvasHeight / sourceImageHeight.toFloat()
        val scale = min(scaleX, scaleY) // Use the smaller scale factor to ensure the image fits

        // Calculate the size of the scaled image within the canvas
        val scaledImageWidth = sourceImageWidth * scale
        val scaledImageHeight = sourceImageHeight * scale

        // Calculate offsets to center the scaled image within the canvas
        val offsetX = (canvasWidth - scaledImageWidth) / 2f
        val offsetY = (canvasHeight - scaledImageHeight) / 2f
        // --- End Scaling Logic ---

        detections.forEach { detection ->
            val rect = detection.rect
            val color = getColorForClass(detection.label)
            val labelId = detection.label
            val confidence = detection.confidence

            // Scale and translate the bounding box coordinates from source image space to canvas space
            val scaledLeft = rect.left * scale + offsetX
            val scaledTop = rect.top * scale + offsetY
            val scaledWidth = rect.width() * scale
            val scaledHeight = rect.height() * scale

            // Ensure coordinates are within canvas bounds (optional, but good practice)
            val canvasLeft = max(0f, scaledLeft)
            val canvasTop = max(0f, scaledTop)
            val canvasRight = min(canvasWidth, scaledLeft + scaledWidth)
            val canvasBottom = min(canvasHeight, scaledTop + scaledHeight)
            val canvasWidthClamped = canvasRight - canvasLeft
            val canvasHeightClamped = canvasBottom - canvasTop

            // Skip drawing if the box is entirely outside or has no size
            if (canvasWidthClamped <= 0 || canvasHeightClamped <= 0) {
                 Log.v(TAG, "Skipping box outside canvas: Label=${labelId}")
                 return@forEach // Continue to next detection
            }

            // Draw the bounding box
            drawRect(
                color = color,
                topLeft = Offset(canvasLeft, canvasTop),
                size = Size(canvasWidthClamped, canvasHeightClamped),
                style = Stroke(width = strokeWidthPx)
            )

            // Prepare text label
            val className = cocoClassNames.getOrElse(labelId) { "Class $labelId" }
            val labelText = "$className: ${"%.2f".format(confidence)}"
            val textStyle = TextStyle(
                color = Color.White,
                fontSize = 14.sp, // Adjust size as needed
                fontWeight = FontWeight.Bold,
                background = color.copy(alpha = 0.7f) // Semi-transparent background
            )

            // Measure the text
            val textLayoutResult: TextLayoutResult = textMeasurer.measure(
                text = AnnotatedString(labelText),
                style = textStyle
            )

            // Calculate position for the text label (above the box)
            val textX = canvasLeft + strokeWidthPx // Small padding from the left edge
            val textY = canvasTop - textLayoutResult.size.height - strokeWidthPx // Position above the box
            val textTopLeft = Offset(textX, max(0f, textY)) // Ensure text isn't drawn above canvas top

            // Draw the text label
            drawText(
                textLayoutResult = textLayoutResult,
                topLeft = textTopLeft
            )
        }
    }
}
