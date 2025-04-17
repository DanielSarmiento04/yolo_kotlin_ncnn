package com.example.yolo_kotlin_ncnn

import android.annotation.SuppressLint
import android.graphics.ImageFormat
import android.util.Log
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer

object ImageUtils {

    private const val TAG = "ImageUtils"

    /**
     * Converts an ImageProxy in YUV_420_888 format to an RGBA ByteArray.
     *
     * IMPORTANT: This assumes the ImageProxy format is YUV_420_888.
     * It allocates a new ByteArray for the RGBA data.
     *
     * @param image The ImageProxy object (must be YUV_420_888 format).
     * @return ByteArray containing RGBA pixel data, or null if conversion fails or format is wrong.
     */
    @SuppressLint("UnsafeOptInUsageError")
    fun imageProxyToRgbaByteArray(image: ImageProxy): ByteArray? {
        if (image.format != ImageFormat.YUV_420_888) {
            Log.e(TAG, "Image format is not YUV_420_888. Got ${image.format}")
            return null
        }

        val width = image.width
        val height = image.height

        val yBuffer = image.planes[0].buffer // Y plane
        val uBuffer = image.planes[1].buffer // U plane (Cb)
        val vBuffer = image.planes[2].buffer // V plane (Cr)

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // Combine Y, U, V planes into NV21 format (Y planar, UV interleaved)
        // This is an intermediate step, NV21 is easier to convert to RGBA
        yBuffer.get(nv21, 0, ySize)

        // Correctly interleave U and V planes into the NV21 buffer
        // NV21 format expects V first, then U in the interleaved plane
        val uvPixelStride = image.planes[1].pixelStride // Should be 2 for YUV_420_888
        val uvRowStride = image.planes[1].rowStride

        // Check if U and V planes are contiguous and pixel stride is 2
        // This is typical for YUV_420_888
        if (uvPixelStride == 2 && uvRowStride == image.planes[2].rowStride && image.planes[1].buffer.remaining() == image.planes[2].buffer.remaining()) {
             // Optimized path for interleaved UV plane (common case)
             // Check which plane comes first in memory
             if (vBuffer.position() > uBuffer.position()) { // V plane data is after U plane data
                 vBuffer.get(nv21, ySize, vSize)
                 uBuffer.get(nv21, ySize + vSize, uSize)
             } else { // U plane data is after V plane data
                 uBuffer.get(nv21, ySize, uSize)
                 vBuffer.get(nv21, ySize + uSize, vSize)
             }
        } else {
            // Slower path for non-contiguous or non-standard UV planes
            val uvBytes = ByteArray(uSize + vSize)
            var uvIndex = 0
            val uRowStride = image.planes[1].rowStride
            val vRowStride = image.planes[2].rowStride

            for (row in 0 until height / 2) {
                for (col in 0 until width / 2) {
                    val uIndex = row * uRowStride + col * uvPixelStride
                    val vIndex = row * vRowStride + col * uvPixelStride

                    if (vIndex < vSize && uIndex < uSize) {
                         // NV21 expects V, U order
                         uvBytes[uvIndex++] = vBuffer.get(vIndex)
                         uvBytes[uvIndex++] = uBuffer.get(uIndex)
                    }
                }
            }
             System.arraycopy(uvBytes, 0, nv21, ySize, uvBytes.size)
        }


        // Allocate RGBA buffer
        val rgbaBytes = ByteArray(width * height * 4)

        // Convert NV21 to RGBA
        var yIndex = 0
        var uvIndex = ySize
        var rgbaIndex = 0

        for (j in 0 until height) {
            for (i in 0 until width) {
                val y = nv21[yIndex++].toInt() and 0xFF
                // Calculate UV indices, considering subsampling (UV are for 2x2 blocks)
                val uvRow = j / 2
                val uvCol = i / 2
                val currentUvIndex = uvIndex + uvRow * width + uvCol * 2 // Index in the combined UV part of nv21

                // NV21 has V, U order
                val v = nv21[currentUvIndex].toInt() and 0xFF
                val u = nv21[currentUvIndex + 1].toInt() and 0xFF

                // YUV to RGB conversion formula
                // Adjust U and V from [0, 255] to [-128, 127] range approx
                val u1 = u - 128
                val v1 = v - 128

                // Calculate R, G, B
                val r = (y + 1.370705f * v1).toInt()
                val g = (y - 0.698001f * v1 - 0.337633f * u1).toInt()
                val b = (y + 1.732446f * u1).toInt()

                // Clamp values to [0, 255] and store in RGBA format
                rgbaBytes[rgbaIndex++] = r.coerceIn(0, 255).toByte()
                rgbaBytes[rgbaIndex++] = g.coerceIn(0, 255).toByte()
                rgbaBytes[rgbaIndex++] = b.coerceIn(0, 255).toByte()
                rgbaBytes[rgbaIndex++] = 255.toByte() // Alpha channel (fully opaque)
            }
        }

        return rgbaBytes
    }
}
