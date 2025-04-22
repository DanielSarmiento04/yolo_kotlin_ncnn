package com.example.yolo_kotlin_ncnn

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

@Composable
fun VulkanStatusChecker(
    isInitialized: Boolean, // Flag indicating if NCNN init succeeded
    isModelLoaded: Boolean, // Flag indicating if model load succeeded
    hasVulkan: Boolean,     // Flag indicating if Vulkan is active
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "NCNN Status",
                style = MaterialTheme.typography.headlineSmall
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Use the passed flags directly
            StatusItem(
                title = "Initialization",
                isSuccess = isInitialized,
                details = if (isInitialized) "NCNN initialized correctly" else "Failed to initialize NCNN"
            )

            StatusItem(
                title = "Model Loading",
                // Model can only be loaded if initialized
                isSuccess = isInitialized && isModelLoaded,
                details = when {
                    !isInitialized -> "Waiting for initialization"
                    isModelLoaded -> "YOLOv11 model loaded"
                    else -> "Model not loaded"
                }
            )

            StatusItem(
                title = "Vulkan Support",
                // Vulkan status is only relevant if initialized
                isSuccess = isInitialized && hasVulkan,
                details = when {
                    !isInitialized -> "Waiting for initialization"
                    hasVulkan -> "Hardware acceleration available (GPU)"
                    else -> "Using CPU fallback (Vulkan not available/enabled)"
                }
            )
        }
    }
}

@Composable
private fun StatusItem(
    title: String,
    isSuccess: Boolean,
    details: String,
    modifier: Modifier = Modifier
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Status indicator
        val statusColor = if (isSuccess) Color.Green else Color.Red
        val statusText = if (isSuccess) "✓" else "✗"
        
        Text(
            text = statusText,
            color = statusColor,
            style = MaterialTheme.typography.bodyLarge
        )
        
        Column(modifier = Modifier.padding(start = 8.dp)) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium
            )
            Text(
                text = details,
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}
