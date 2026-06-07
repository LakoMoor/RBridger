package com.lakomoor.rbridger.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColors = darkColorScheme(
    primary = Color(0xFF9C86E8),
    onPrimary = Color(0xFF1A1A2E),
    primaryContainer = Color(0xFF3D3060),
    background = Color(0xFF121212),
    surface = Color(0xFF1E1E2E),
    onBackground = Color(0xFFE0E0E0),
    onSurface = Color(0xFFE0E0E0),
)

@Composable
fun RBridgerTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = DarkColors, content = content)
}
