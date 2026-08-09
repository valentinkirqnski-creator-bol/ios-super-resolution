package ventures.spacetree.fuzeframe

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/** Progress sink the native side calls back into. */
interface ProgressSink {
    fun onProgress(stage: String, fraction: Float)
}

object Native {
    init { System.loadLibrary("fuzeframe") }

    /** Returns null on success, or an error message. */
    external fun process(dngPaths: Array<String>, outDngPath: String, cb: ProgressSink?): String?

    /** Merged DNG -> packed ARGB_8888. outWH receives width and height. */
    external fun renderArgb(dngPath: String, outWH: IntArray): IntArray?
}

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { Screen() } }
    }

    @Composable
    private fun Screen() {
        var status by remember { mutableStateOf("Pick 2 or more DNG files from the same burst.") }
        var busy by remember { mutableStateOf(false) }
        var result by remember { mutableStateOf<Bitmap?>(null) }
        var savedTo by remember { mutableStateOf<List<String>>(emptyList()) }

        val picker = rememberLauncherForActivityResult(
            ActivityResultContracts.OpenMultipleDocuments()
        ) { uris: List<Uri> ->
            if (uris.size < 2) {
                status = "Pick at least 2 DNG files"
                return@rememberLauncherForActivityResult
            }
            busy = true
            result = null
            savedTo = emptyList()
            lifecycleScope.launch {
                // Status arrives from the worker thread; hop to the UI thread
                // rather than mutate Compose state off it.
                val out = withContext(Dispatchers.Default) {
                    runBurst(uris) { s -> runOnUiThread { status = s } }
                }
                status = out.message
                result = out.bitmap
                savedTo = out.savedTo
                busy = false
            }
        }

        Column(
            Modifier.fillMaxSize().padding(20.dp).verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("FuzeFrame", style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.height(4.dp))
            Text(
                "Handheld multi-frame super-resolution. This build merges DNG files " +
                    "you already have; it does not capture.",
                style = MaterialTheme.typography.bodySmall
            )
            Spacer(Modifier.height(16.dp))
            Button(onClick = { picker.launch(arrayOf("*/*")) }, enabled = !busy) {
                Text(if (busy) "Working…" else "Choose DNG files")
            }
            Spacer(Modifier.height(12.dp))
            if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
            Spacer(Modifier.height(12.dp))
            Text(status, style = MaterialTheme.typography.bodyMedium)
            savedTo.forEach {
                Spacer(Modifier.height(4.dp))
                Text("Saved: $it", style = MaterialTheme.typography.bodySmall)
            }
            result?.let {
                Spacer(Modifier.height(16.dp))
                Image(it.asImageBitmap(), null, Modifier.fillMaxWidth())
            }
        }
    }

    private class Outcome(val message: String, val bitmap: Bitmap?, val savedTo: List<String>)

    private fun runBurst(uris: List<Uri>, onStatus: (String) -> Unit): Outcome {
        // SAF gives content:// URIs; LibRaw needs real paths, so each frame is
        // copied into cache first. Deleted immediately after -- a burst of 13MP
        // DNGs is easily several hundred megabytes.
        val work = File(cacheDir, "burst").apply { deleteRecursively(); mkdirs() }
        val outDir = getExternalFilesDir(null) ?: cacheDir
        val stamp = System.currentTimeMillis()
        val outDng = File(outDir, "fuzeframe_$stamp.dng")
        val outJpg = File(outDir, "fuzeframe_$stamp.jpg")
        try {
            val paths = ArrayList<String>(uris.size)
            uris.forEachIndexed { i, uri ->
                val f = File(work, "frame_%02d.dng".format(i))
                contentResolver.openInputStream(uri)?.use { input ->
                    f.outputStream().use { input.copyTo(it) }
                } ?: return Outcome("Could not read a selected file", null, emptyList())
                paths.add(f.absolutePath)
            }

            val sink = object : ProgressSink {
                override fun onProgress(stage: String, fraction: Float) {
                    onStatus("%s  %d%%".format(stage, (fraction * 100).toInt()))
                }
            }
            val err = Native.process(paths.toTypedArray(), outDng.absolutePath, sink)
            if (err != null) return Outcome(err, null, emptyList())

            // The input copies are no longer needed and the render is the
            // second-largest allocation in the run; free the disk first.
            work.deleteRecursively()

            onStatus("Rendering JPEG")
            val wh = IntArray(2)
            val px = Native.renderArgb(outDng.absolutePath, wh)
                ?: return Outcome("Merged to DNG, but rendering the JPEG failed",
                                  null, listOf(outDng.absolutePath))

            val w = wh[0]
            val h = wh[1]
            val full = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
            // 82 to match the iOS export, where it was measured at 46% of the
            // size of quality 92 for 2.8 LSB RMS.
            outJpg.outputStream().use { full.compress(Bitmap.CompressFormat.JPEG, 82, it) }

            // Downscale for display; the full bitmap is ~50MB at 13MP and there
            // is no reason to hold it alive behind a preview.
            val scale = 1400f / maxOf(w, h)
            val shown = if (scale < 1f)
                Bitmap.createScaledBitmap(full, (w * scale).toInt(), (h * scale).toInt(), true)
            else full
            if (shown !== full) full.recycle()

            return Outcome("Done — %d frames, %d x %d".format(uris.size, w, h),
                           shown, listOf(outDng.absolutePath, outJpg.absolutePath))
        } catch (t: Throwable) {
            return Outcome("Failed: ${t.message}", null, emptyList())
        } finally {
            work.deleteRecursively()
        }
    }
}
