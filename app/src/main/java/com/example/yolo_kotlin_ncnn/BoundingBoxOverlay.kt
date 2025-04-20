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

private const val TAG = "BoundingBoxOverlay" // Logging tag

// Simple color cache for classes
private val classColors = mutableMapOf<Int, Color>()
private fun getColorForClass(classId: Int): Color {
    return classColors.getOrPut(classId) {
        // Generate visually distinct colors if possible
        val hue = (classId * 60 + 30) % 360 // Spread hues, add offset
        val saturation = 0.7f + (classId % 5) * 0.06f // Vary saturation slightly
        val value = 0.8f + (classId % 4) * 0.05f // Vary value slightly
        Color.hsv(hue = hue.toFloat(), saturation = saturation, value = value, alpha = 1.0f)
    }
}

// COCO class names + Custom Classes (ensure this matches the NUM_CLASSES in native-lib.cpp)
// Verify this list matches the classes your specific YOLOv11 model was trained on.
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
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", // 80 classes for standard COCO
    // Custom classes from model.yml (indices 80-83)
    "pump", "pipe", "steel pipe", "electric cable"
) // Total 84 classes

@Composable
fun BoundingBoxOverlay(
    detections: List<Detection>, // Updated type to use the defined Detection data class
    sourceImageWidth: Int,
    sourceImageHeight: Int,
    modifier: Modifier = Modifier
) {
    val density = LocalDensity.current
    val strokeWidthPx = with(density) { 2.dp.toPx() } // Stroke width for boxes
    val textMeasurer = rememberTextMeasurer()

    Canvas(modifier = modifier.fillMaxSize()) {
        // Check for valid dimensions before proceeding
        if (sourceImageWidth <= 0 || sourceImageHeight <= 0) {
            // Log only once or less frequently if this state persists
            // Log.w(TAG, "Skipping draw: Invalid source image dimensions ($sourceImageWidth x $sourceImageHeight)")
            return@Canvas
        }

        val canvasWidth = size.width
        val canvasHeight = size.height

        // --- Scaling Logic ---
        // Assumes the camera preview uses scale type FIT_CENTER.
        // Scales the source image dimensions to fit within the canvas while maintaining aspect ratio,
        // then calculates the offset to center the scaled image.
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

        // Log scaling factors once if dimensions change significantly (optional)
        // Log.d(TAG, "Canvas: ${canvasWidth}x${canvasHeight}, Source: ${sourceImageWidth}x${sourceImageHeight}, Scale: $scale, Offset: ${offsetX}x${offsetY}")

        detections.forEach { detection ->
            val rect = detection.rect // Bounding box in original image coordinates
            val color = getColorForClass(detection.label)
            val labelId = detection.label
            val confidence = detection.confidence

            // Scale and translate the bounding box coordinates from source image space to canvas space
            val scaledLeft = rect.left * scale + offsetX
            val scaledTop = rect.top * scale + offsetY
            val scaledWidth = rect.width() * scale
            val scaledHeight = rect.height() * scale

            // Ensure coordinates are within canvas bounds after scaling and offset
            val canvasLeft = max(0f, scaledLeft)
            val canvasTop = max(0f, scaledTop)
            // Calculate right/bottom based on scaled dimensions before clamping top/left
            val canvasRight = min(canvasWidth, scaledLeft + scaledWidth)
            val canvasBottom = min(canvasHeight, scaledTop + scaledHeight)
            // Calculate clamped width/height based on clamped coordinates
            val canvasWidthClamped = canvasRight - canvasLeft
            val canvasHeightClamped = canvasBottom - canvasTop

            // Skip drawing if the box is entirely outside the canvas or has zero/negative size after clamping
            if (canvasWidthClamped <= strokeWidthPx / 2 || canvasHeightClamped <= strokeWidthPx / 2) {
                 // Log.v(TAG, "Skipping box outside canvas or too small: Label=${labelId}") // Verbose log
                 return@forEach // Continue to next detection
            }

            // Draw the bounding box
            drawRect(
                color = color,
                topLeft = Offset(canvasLeft, canvasTop),
                size = Size(canvasWidthClamped, canvasHeightClamped),
                style = Stroke(width = strokeWidthPx)
            )

            // Prepare text label using class names list
            val className = cocoClassNames.getOrElse(labelId) { "ID $labelId" } // Fallback to ID if name not found
            val labelText = "$className: ${"%.2f".format(confidence)}"
            val textStyle = TextStyle(
                color = Color.White,
                fontSize = 14.sp, // Adjust size as needed
                fontWeight = FontWeight.Bold,
                background = color.copy(alpha = 0.7f) // Semi-transparent background matching box color
            )

            // Measure the text
            val textLayoutResult: TextLayoutResult = textMeasurer.measure(
                text = AnnotatedString(labelText),
                style = textStyle,
                maxLines = 1 // Ensure text stays on one line
            )

            // Calculate position for the text label (prefer above the box, move inside if needed)
            val textWidth = textLayoutResult.size.width
            val textHeight = textLayoutResult.size.height

            // Position text slightly inside the box from the top-left corner
            var textX = canvasLeft + strokeWidthPx
            var textY = canvasTop + strokeWidthPx

            // Optional: Try to place text above the box if there's space
            val potentialTextYAbove = canvasTop - textHeight - strokeWidthPx
            if (potentialTextYAbove >= 0) { // Check if it fits above within canvas bounds
                 textY = potentialTextYAbove
            }

            // Ensure text doesn't go outside the right edge of the canvas
            if (textX + textWidth > canvasWidth) {
                textX = canvasWidth - textWidth - strokeWidthPx // Adjust to fit
            }
            // Ensure text doesn't go outside the bottom edge (if placed inside) or top edge (if placed above)
            textY = textY.coerceIn(0f, canvasHeight - textHeight)


            // Draw the text label
            drawText(
                textLayoutResult = textLayoutResult,
                topLeft = Offset(textX, textY)
            )
        }
    }
}
