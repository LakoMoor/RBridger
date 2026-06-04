package com.lakomoor.rbridger

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

object ModelManager {
    private const val MODEL_URL =
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    private const val MODEL_FILENAME = "face_landmarker.task"
    private const val MIN_VALID_SIZE = 1_000_000L

    suspend fun ensureModel(context: Context, onProgress: (Float) -> Unit): File =
        withContext(Dispatchers.IO) {
            val file = File(context.filesDir, MODEL_FILENAME)
            if (file.exists() && file.length() > MIN_VALID_SIZE) return@withContext file

            val conn = URL(MODEL_URL).openConnection() as HttpURLConnection
            conn.connect()
            val total = conn.contentLengthLong
            var downloaded = 0L

            conn.inputStream.use { input ->
                file.outputStream().use { output ->
                    val buf = ByteArray(16 * 1024)
                    var n: Int
                    while (input.read(buf).also { n = it } != -1) {
                        output.write(buf, 0, n)
                        downloaded += n
                        if (total > 0) onProgress(downloaded.toFloat() / total)
                    }
                }
            }
            conn.disconnect()
            file
        }
}
