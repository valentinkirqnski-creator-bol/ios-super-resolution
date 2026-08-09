package ventures.spacetree.fuzeframe

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.net.Uri
import android.os.Bundle
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
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
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.CompletableDeferred
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

    private val engine by lazy { CameraEngine(this) }
    private var caps: CameraEngine.Caps? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        caps = engine.probe()
        setContent { MaterialTheme { Screen() } }
    }

    override fun onDestroy() {
        engine.stop()
        super.onDestroy()
    }

    @Composable
    private fun Screen() {
        val c = caps
        var granted by remember {
            mutableStateOf(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED)
        }
        var frames by remember { mutableStateOf(4) }   // iOS default
        var status by remember {
            mutableStateOf(
                if (c?.hasRaw == true) "Hold steady and press the shutter."
                else "This device does not report Camera2 RAW support."
            )
        }
        var busy by remember { mutableStateOf(false) }
        var result by remember { mutableStateOf<Bitmap?>(null) }
        var savedTo by remember { mutableStateOf<List<String>>(emptyList()) }
        var showReport by remember { mutableStateOf(false) }

        val permission = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { ok -> granted = ok; if (!ok) status = "Camera permission denied" }

        LaunchedEffect(Unit) {
            if (!granted && c?.hasRaw == true) permission.launch(Manifest.permission.CAMERA)
        }

        val picker = rememberLauncherForActivityResult(
            ActivityResultContracts.OpenMultipleDocuments()
        ) { uris: List<Uri> ->
            if (uris.size < 2) {
                status = "Pick at least 2 DNG files"
                return@rememberLauncherForActivityResult
            }
            busy = true; result = null; savedTo = emptyList()
            lifecycleScope.launch {
                val out = withContext(Dispatchers.Default) {
                    val dir = File(cacheDir, "burst").apply { deleteRecursively(); mkdirs() }
                    val files = copyIn(uris, dir)
                    if (files == null) Outcome("Could not read a selected file", null, emptyList())
                    else merge(files) { s -> runOnUiThread { status = s } }
                }
                status = out.message; result = out.bitmap; savedTo = out.savedTo; busy = false
            }
        }

        Column(
            Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("FuzeFrame", style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.height(12.dp))

            // `c != null && c.hasRaw` rather than `c?.hasRaw == true`: the latter
            // does not smart-cast c, and the calls below need it non-null.
            if (c != null && c.hasRaw && granted) {
                val preview = remember(c) { engine.previewSize(c) }
                val rotation = remember(c) { engine.sensorOrientation(c) }
                AndroidView(
                    factory = { ctx ->
                        TextureView(ctx).apply {
                            layoutParams = ViewGroup.LayoutParams(
                                ViewGroup.LayoutParams.MATCH_PARENT,
                                ViewGroup.LayoutParams.MATCH_PARENT)
                            surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                                override fun onSurfaceTextureAvailable(
                                    st: SurfaceTexture, w: Int, h: Int) {
                                    st.setDefaultBufferSize(preview.width, preview.height)
                                    applyPreviewTransform(this@apply, preview, rotation, w, h)
                                    try {
                                        engine.start(c, Surface(st),
                                            ready = { runOnUiThread { status = "Ready" } },
                                            fail = { e -> runOnUiThread { status = e } })
                                    } catch (t: SecurityException) {
                                        runOnUiThread { status = "Camera permission denied" }
                                    }
                                }
                                override fun onSurfaceTextureSizeChanged(
                                    st: SurfaceTexture, w: Int, h: Int) {
                                    applyPreviewTransform(this@apply, preview, rotation, w, h)
                                }
                                override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
                                    engine.stop(); return true
                                }
                                override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
                            }
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        // The preview buffer is landscape (sensor orientation) while
                        // the activity is locked portrait, so the displayed aspect is
                        // the inverse of the buffer's.
                        .aspectRatio(preview.height.toFloat() / preview.width)
                )
                Spacer(Modifier.height(12.dp))

                Text("Frames: $frames", style = MaterialTheme.typography.bodyMedium)
                Slider(
                    value = frames.toFloat(),
                    onValueChange = { frames = it.toInt() },
                    valueRange = 2f..8f,
                    steps = 5,
                    enabled = !busy
                )
                Text(
                    "Every stage runs on the CPU here, so the merge takes tens of " +
                        "seconds. Start low.",
                    style = MaterialTheme.typography.bodySmall
                )
                Spacer(Modifier.height(12.dp))

                Button(
                    onClick = {
                        busy = true; result = null; savedTo = emptyList()
                        lifecycleScope.launch {
                            val out = shootAndMerge(frames) { s -> runOnUiThread { status = s } }
                            status = out.message; result = out.bitmap
                            savedTo = out.savedTo; busy = false
                            engine.unlock()
                        }
                    },
                    enabled = !busy,
                    modifier = Modifier.fillMaxWidth()
                ) { Text(if (busy) "Working…" else "Shutter") }
            } else {
                Text(
                    if (c?.hasRaw == true) "Camera permission is needed to capture a burst."
                    else "No camera on this device reports RAW (DNG) capture, so FuzeFrame " +
                         "cannot shoot its own burst here. You can still merge DNG files " +
                         "captured elsewhere.",
                    style = MaterialTheme.typography.bodyMedium
                )
                if (c?.hasRaw == true && !granted) {
                    Spacer(Modifier.height(8.dp))
                    Button(onClick = { permission.launch(Manifest.permission.CAMERA) }) {
                        Text("Grant camera access")
                    }
                }
                Spacer(Modifier.height(8.dp))
                TextButton(onClick = { showReport = !showReport }) {
                    Text(if (showReport) "Hide camera report" else "What does my device report?")
                }
                if (showReport) {
                    Text(c?.report ?: "no camera information",
                         style = MaterialTheme.typography.bodySmall)
                }
            }

            Spacer(Modifier.height(12.dp))
            OutlinedButton(onClick = { picker.launch(arrayOf("*/*")) }, enabled = !busy) {
                Text("Merge DNG files instead")
            }

            Spacer(Modifier.height(12.dp))
            if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
            Spacer(Modifier.height(8.dp))
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

    /**
     * Camera2 delivers preview frames in sensor orientation, which on a back
     * camera is 90 degrees off a portrait-locked activity. TextureView will not
     * correct that on its own, so the buffer is rotated about the view centre and
     * scaled up to cover the shorter axis it now spans.
     */
    private fun applyPreviewTransform(
        view: TextureView, buffer: Size, sensorOrientation: Int, viewW: Int, viewH: Int
    ) {
        if (viewW <= 0 || viewH <= 0) return
        val m = Matrix()
        val cx = viewW / 2f
        val cy = viewH / 2f
        // Activity is locked portrait, so the display contributes no rotation.
        val degrees = ((sensorOrientation % 360) + 360) % 360
        m.postRotate(-degrees.toFloat(), cx, cy)
        if (degrees == 90 || degrees == 270) {
            // After a quarter turn the buffer's long axis lies across the view's
            // short one; scale so it still covers.
            val scale = maxOf(viewW.toFloat() / viewH, viewH.toFloat() / viewW)
            m.postScale(scale, scale, cx, cy)
        }
        view.setTransform(m)
    }

    /** Captures a burst, then merges it. */
    private suspend fun shootAndMerge(frames: Int, onStatus: (String) -> Unit): Outcome {
        val dir = File(cacheDir, "burst").apply { deleteRecursively(); mkdirs() }
        val shot = CompletableDeferred<Pair<List<File>?, String?>>()
        engine.shoot(frames, dir,
            onStatus = { s -> onStatus(s) },
            onDone = { files, err -> shot.complete(files to err) })
        val (files, err) = shot.await()
        if (files == null || files.size < 2) {
            return Outcome(err ?: "Captured too few frames to merge", null, emptyList())
        }
        return withContext(Dispatchers.Default) { merge(files, onStatus) }
    }

    private fun copyIn(uris: List<Uri>, dir: File): List<File>? {
        val out = ArrayList<File>(uris.size)
        uris.forEachIndexed { i, uri ->
            val f = File(dir, "frame_%02d.dng".format(i))
            contentResolver.openInputStream(uri)?.use { input ->
                f.outputStream().use { input.copyTo(it) }
            } ?: return null
            out.add(f)
        }
        return out
    }

    /** Native merge + ISP render + JPEG encode. Runs off the main thread. */
    private fun merge(frames: List<File>, onStatus: (String) -> Unit): Outcome {
        val outDir = getExternalFilesDir(null) ?: cacheDir
        val stamp = System.currentTimeMillis()
        val outDng = File(outDir, "fuzeframe_$stamp.dng")
        val outJpg = File(outDir, "fuzeframe_$stamp.jpg")
        val work = frames.firstOrNull()?.parentFile
        try {
            val sink = object : ProgressSink {
                override fun onProgress(stage: String, fraction: Float) {
                    onStatus("%s  %d%%".format(stage, (fraction * 100).toInt()))
                }
            }
            val err = Native.process(
                frames.map { it.absolutePath }.toTypedArray(), outDng.absolutePath, sink)
            if (err != null) return Outcome(err, null, emptyList())

            // The captured frames are no longer needed and the render is the
            // second-largest allocation in the run; free the disk first.
            work?.deleteRecursively()

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

            return Outcome("Done — %d frames, %d x %d".format(frames.size, w, h),
                           shown, listOf(outDng.absolutePath, outJpg.absolutePath))
        } catch (t: Throwable) {
            return Outcome("Failed: ${t.message}", null, emptyList())
        } finally {
            work?.deleteRecursively()
        }
    }
}
