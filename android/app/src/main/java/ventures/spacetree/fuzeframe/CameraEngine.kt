package ventures.spacetree.fuzeframe

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.DngCreator
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.Surface
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executor

/**
 * Camera2 RAW burst capture, mirroring the iOS capture policy in CameraModel.swift:
 * focus, white balance and exposure are locked for the whole burst, flash off, no
 * stabilisation, and every frame is written as a DNG that the native pipeline reads
 * exactly as it reads an imported one.
 *
 * The one iOS feature not reproduced here is the zero-shutter-lag ring. Holding a
 * rolling buffer of RAW frames costs 26MB each on a 13MP sensor, which is not a
 * trade this class of device can make; the shutter therefore starts the burst
 * rather than reaching backwards from it.
 */
class CameraEngine(private val ctx: Context) {

    /** What the device actually advertises. Null cameraId means no RAW anywhere. */
    class Caps(
        val cameraId: String?,
        val rawSize: Size?,
        val manualSensor: Boolean,
        val hardwareLevel: Int,
        val report: String
    ) {
        val hasRaw: Boolean get() = cameraId != null && rawSize != null
    }

    private val mgr = ctx.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private val executor = Executor { r -> handler?.post(r) ?: r.run() }

    private var device: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var chars: CameraCharacteristics? = null
    private var previewSurface: Surface? = null

    @Volatile private var lastResult: TotalCaptureResult? = null

    // A burst in flight. Images and their results arrive on different callbacks
    // and are matched on the sensor timestamp, which is the only field both carry.
    private val images = HashMap<Long, Image>()
    private val results = HashMap<Long, TotalCaptureResult>()
    private var burstDir: File? = null
    private var burstWanted = 0
    private val burstFiles = ArrayList<File>()
    private var burstStatus: ((String) -> Unit)? = null
    private var burstDone: ((List<File>?, String?) -> Unit)? = null

    // ---------------------------------------------------------------- capability

    /**
     * Picks the back camera that can emit RAW_SENSOR. Also builds a human-readable
     * report: on a device with no RAW support that report is the only useful thing
     * the app can say, so it is worth assembling even on the failure path.
     */
    fun probe(): Caps {
        val sb = StringBuilder()
        var bestId: String? = null
        var bestSize: Size? = null
        var manual = false
        var level = -1
        try {
            for (id in mgr.cameraIdList) {
                val c = mgr.getCameraCharacteristics(id)
                val facing = c.get(CameraCharacteristics.LENS_FACING)
                val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
                val hw = c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL) ?: -1
                val raw = caps.contains(
                    CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW)
                val man = caps.contains(
                    CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR)
                val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                val sizes = map?.getOutputSizes(ImageFormat.RAW_SENSOR)
                val largest = sizes?.maxByOrNull { it.width.toLong() * it.height }

                sb.append("camera $id: ${facingName(facing)}, ${levelName(hw)}")
                sb.append(if (raw) ", RAW" else ", no RAW")
                if (man) sb.append(", manual")
                if (largest != null) sb.append(", ${largest.width}x${largest.height}")
                sb.append('\n')

                val isBack = facing == CameraCharacteristics.LENS_FACING_BACK
                if (raw && largest != null && bestId == null && isBack) {
                    bestId = id; bestSize = largest; manual = man; level = hw
                }
            }
            // A front camera with RAW is still better than nothing.
            if (bestId == null) {
                for (id in mgr.cameraIdList) {
                    val c = mgr.getCameraCharacteristics(id)
                    val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: IntArray(0)
                    if (!caps.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW)) continue
                    val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
                    val largest = map?.getOutputSizes(ImageFormat.RAW_SENSOR)
                        ?.maxByOrNull { it.width.toLong() * it.height } ?: continue
                    bestId = id; bestSize = largest
                    manual = caps.contains(
                        CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR)
                    level = c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL) ?: -1
                    break
                }
            }
        } catch (e: Throwable) {
            sb.append("camera query failed: ${e.message}\n")
        }
        return Caps(bestId, bestSize, manual, level, sb.toString().trim())
    }

    private fun facingName(f: Int?) = when (f) {
        CameraCharacteristics.LENS_FACING_BACK -> "back"
        CameraCharacteristics.LENS_FACING_FRONT -> "front"
        else -> "external"
    }

    private fun levelName(l: Int) = when (l) {
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY -> "LEGACY"
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LIMITED -> "LIMITED"
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL -> "FULL"
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_3 -> "LEVEL_3"
        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_EXTERNAL -> "EXTERNAL"
        else -> "unknown"
    }

    /** Degrees the sensor output is rotated relative to the device's natural orientation. */
    fun sensorOrientation(caps: Caps): Int {
        val id = caps.cameraId ?: return 90
        return try {
            mgr.getCameraCharacteristics(id).get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
        } catch (_: Throwable) { 90 }
    }

    /** Largest preview size at or under 1080p whose aspect matches the sensor. */
    fun previewSize(caps: Caps): Size {
        val c = caps.cameraId?.let { mgr.getCameraCharacteristics(it) } ?: return Size(1280, 720)
        val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?: return Size(1280, 720)
        val target = caps.rawSize?.let { it.width.toFloat() / it.height } ?: (4f / 3f)
        val options = map.getOutputSizes(android.view.SurfaceHolder::class.java) ?: return Size(1280, 720)
        return options
            .filter { it.width <= 1920 && it.height <= 1920 }
            .minByOrNull {
                val ar = it.width.toFloat() / it.height
                // Aspect first, then prefer the larger surface.
                Math.abs(ar - target) * 1000f - (it.width * it.height) / 1_000_000f
            } ?: Size(1280, 720)
    }

    // ---------------------------------------------------------------- lifecycle

    // The callbacks are named `ready`/`fail` rather than onReady/onError on
    // purpose: CameraDevice.StateCallback has its own onError member, and a
    // captured lambda of the same name inside that object expression is exactly
    // the kind of shadowing that resolves to the wrong thing.
    @Throws(SecurityException::class)
    fun start(caps: Caps, preview: Surface, ready: () -> Unit, fail: (String) -> Unit) {
        if (!caps.hasRaw) { fail("This device does not expose RAW capture"); return }
        stop()
        previewSurface = preview
        val t = HandlerThread("fuzeframe-cam").apply { start() }
        thread = t
        handler = Handler(t.looper)

        val id = caps.cameraId!!
        val size = caps.rawSize!!
        chars = mgr.getCameraCharacteristics(id)
        // 3 in flight: each RAW frame is ~26MB at 13MP, and they are drained to
        // disk as fast as they arrive, so a deeper queue only raises the peak.
        reader = ImageReader.newInstance(size.width, size.height, ImageFormat.RAW_SENSOR, 3)
        reader!!.setOnImageAvailableListener({ r ->
            val img = try { r.acquireNextImage() } catch (t2: Throwable) { null } ?: return@setOnImageAvailableListener
            synchronized(this) {
                images[img.timestamp] = img
                drainPairs()
            }
        }, handler)

        try {
            mgr.openCamera(id, object : CameraDevice.StateCallback() {
                override fun onOpened(cam: CameraDevice) {
                    device = cam
                    configureSession(cam, ready, fail)
                }
                override fun onDisconnected(cam: CameraDevice) { cam.close(); device = null }
                override fun onError(cam: CameraDevice, error: Int) {
                    cam.close(); device = null
                    fail("Camera error $error")
                }
            }, handler)
        } catch (e: CameraAccessException) {
            fail("Cannot open camera: ${e.message}")
        }
    }

    private fun configureSession(cam: CameraDevice, ready: () -> Unit, fail: (String) -> Unit) {
        val preview = previewSurface ?: return fail("No preview surface")
        val raw = reader?.surface ?: return fail("No RAW surface")
        val outs = listOf(OutputConfiguration(preview), OutputConfiguration(raw))
        val cb = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(s: CameraCaptureSession) {
                session = s
                startPreview(cam, s, fail)
                ready()
            }
            override fun onConfigureFailed(s: CameraCaptureSession) {
                fail("Could not configure the camera for RAW + preview")
            }
        }
        try {
            cam.createCaptureSession(
                SessionConfiguration(SessionConfiguration.SESSION_REGULAR, outs, executor, cb))
        } catch (e: Throwable) {
            fail("Session setup failed: ${e.message}")
        }
    }

    private fun startPreview(cam: CameraDevice, s: CameraCaptureSession, fail: (String) -> Unit) {
        val preview = previewSurface ?: return
        try {
            val b = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            b.addTarget(preview)
            b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
            b.set(CaptureRequest.CONTROL_AF_MODE,
                  CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            b.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF)
            s.setRepeatingRequest(b.build(), object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(
                    sess: CameraCaptureSession, req: CaptureRequest, result: TotalCaptureResult) {
                    lastResult = result
                }
            }, handler)
        } catch (e: Throwable) {
            fail("Preview failed: ${e.message}")
        }
    }

    fun stop() {
        try { session?.close() } catch (_: Throwable) {}
        try { device?.close() } catch (_: Throwable) {}
        try { reader?.close() } catch (_: Throwable) {}
        session = null; device = null; reader = null
        synchronized(this) {
            images.values.forEach { try { it.close() } catch (_: Throwable) {} }
            images.clear(); results.clear()
        }
        thread?.quitSafely()
        thread = null; handler = null
    }

    // ---------------------------------------------------------------- capture

    /**
     * Locks AE/AWB/AF, then captures [frames] identical exposures.
     *
     * Identical is the point: the merge assumes every frame in the burst sees the
     * same scene brightness, and an auto-exposure that drifts mid-burst shows up
     * as a global brightness step the robustness mask reads as motion. Where the
     * device supports MANUAL_SENSOR the exposure is pinned numerically; where it
     * does not, AE lock is the best available substitute.
     */
    fun shoot(
        frames: Int,
        outDir: File,
        onStatus: (String) -> Unit,
        onDone: (List<File>?, String?) -> Unit
    ) {
        val cam = device
        val s = session
        val raw = reader?.surface
        if (cam == null || s == null || raw == null) { onDone(null, "Camera is not running"); return }

        outDir.deleteRecursively(); outDir.mkdirs()
        synchronized(this) {
            images.values.forEach { try { it.close() } catch (_: Throwable) {} }
            images.clear(); results.clear(); burstFiles.clear()
            burstDir = outDir
            burstWanted = frames
            burstStatus = onStatus
            burstDone = onDone
        }

        onStatus("Locking exposure and focus")
        lockThenCapture(cam, s, raw, frames, onStatus, onDone)
    }

    private fun lockThenCapture(
        cam: CameraDevice,
        s: CameraCaptureSession,
        raw: Surface,
        frames: Int,
        onStatus: (String) -> Unit,
        onDone: (List<File>?, String?) -> Unit
    ) {
        val preview = previewSurface ?: return onDone(null, "No preview surface")
        try {
            // Kick AF and AE once, then poll the repeating result for convergence.
            val b = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            b.addTarget(preview)
            b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            b.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START)
            b.set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER,
                  CameraMetadata.CONTROL_AE_PRECAPTURE_TRIGGER_START)
            s.capture(b.build(), null, handler)
        } catch (_: Throwable) {
            // Not fatal: a LIMITED device may reject the trigger, and AE lock below
            // still does the important half of the job.
        }

        // Convergence is polled with a hard deadline. Waiting on a state machine a
        // vendor may never drive to CONVERGED is the classic way a camera app hangs
        // on exactly the hardware you most need it to work on.
        val deadline = System.currentTimeMillis() + 2000
        val poll = object : Runnable {
            override fun run() {
                val r = lastResult
                val ae = r?.get(CaptureResult.CONTROL_AE_STATE)
                val af = r?.get(CaptureResult.CONTROL_AF_STATE)
                val aeReady = ae == null ||
                    ae == CaptureResult.CONTROL_AE_STATE_CONVERGED ||
                    ae == CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED ||
                    ae == CaptureResult.CONTROL_AE_STATE_LOCKED
                val afReady = af == null ||
                    af == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED ||
                    af == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED ||
                    af == CaptureResult.CONTROL_AF_STATE_INACTIVE
                if ((aeReady && afReady) || System.currentTimeMillis() > deadline) {
                    fireBurst(cam, s, raw, frames, onStatus, onDone)
                } else {
                    handler?.postDelayed(this, 50)
                }
            }
        }
        handler?.post(poll)
    }

    private fun fireBurst(
        cam: CameraDevice,
        s: CameraCaptureSession,
        raw: Surface,
        frames: Int,
        onStatus: (String) -> Unit,
        onDone: (List<File>?, String?) -> Unit
    ) {
        val c = chars
        val converged = lastResult
        val expTime = converged?.get(CaptureResult.SENSOR_EXPOSURE_TIME)
        val iso = converged?.get(CaptureResult.SENSOR_SENSITIVITY)
        val manual = c?.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
            ?.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true

        try {
            // Hold the preview steady on locked settings while the burst runs, so
            // the viewfinder does not re-meter against the frames being captured.
            previewSurface?.let { preview ->
                val pb = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                pb.addTarget(preview)
                pb.set(CaptureRequest.CONTROL_AE_LOCK, true)
                pb.set(CaptureRequest.CONTROL_AWB_LOCK, true)
                pb.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_IDLE)
                s.setRepeatingRequest(pb.build(),
                    object : CameraCaptureSession.CaptureCallback() {
                        override fun onCaptureCompleted(
                            sess: CameraCaptureSession, req: CaptureRequest, r: TotalCaptureResult) {
                            lastResult = r
                        }
                    }, handler)
            }

            val b = cam.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
            b.addTarget(raw)
            b.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF)
            b.set(CaptureRequest.CONTROL_AWB_LOCK, true)
            b.set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_IDLE)
            if (manual && expTime != null && iso != null) {
                b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
                b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, expTime)
                b.set(CaptureRequest.SENSOR_SENSITIVITY, iso)
                b.set(CaptureRequest.SENSOR_FRAME_DURATION, minFrameDuration(expTime))
            } else {
                b.set(CaptureRequest.CONTROL_AE_LOCK, true)
            }
            val req = b.build()
            val list = ArrayList<CaptureRequest>(frames)
            repeat(frames) { list.add(req) }

            if (expTime != null && iso != null) {
                onStatus("Capturing %d frames  1/%d s  ISO %d"
                    .format(frames, Math.max(1L, 1_000_000_000L / expTime), iso))
            } else {
                onStatus("Capturing $frames frames")
            }

            s.captureBurst(list, object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(
                    sess: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                    val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
                    synchronized(this@CameraEngine) {
                        results[ts] = result
                        drainPairs()
                    }
                }
                override fun onCaptureFailed(
                    sess: CameraCaptureSession, request: CaptureRequest,
                    failure: android.hardware.camera2.CaptureFailure) {
                    synchronized(this@CameraEngine) {
                        // One dropped frame is survivable; the burst simply merges
                        // fewer. Lower the target so the pipeline is not waiting
                        // forever for an image that will never arrive.
                        burstWanted -= 1
                        drainPairs()
                    }
                }
            }, handler)
        } catch (e: Throwable) {
            onDone(null, "Capture failed: ${e.message}")
        }
    }

    private fun minFrameDuration(exposureNs: Long): Long {
        val c = chars ?: return exposureNs
        val map: StreamConfigurationMap =
            c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: return exposureNs
        val size = reader?.let { Size(it.width, it.height) } ?: return exposureNs
        val min = try {
            map.getOutputMinFrameDuration(ImageFormat.RAW_SENSOR, size)
        } catch (_: Throwable) { 0L }
        // The frame duration must cover the exposure, or the request is rejected.
        return Math.max(min, exposureNs)
    }

    /** Must hold the monitor. Writes every image whose result has also arrived. */
    private fun drainPairs() {
        val c = chars ?: return
        val dir = burstDir ?: return
        val it = images.entries.iterator()
        while (it.hasNext()) {
            val e = it.next()
            val key = e.key            // read before remove(); the entry is dead after
            val result = results[key] ?: continue
            val img: Image = e.value
            it.remove()
            results.remove(key)
            val f = File(dir, "frame_%02d.dng".format(burstFiles.size))
            try {
                // Explicit close, not use{}: DngCreator implements AutoCloseable
                // but not Closeable, which is what kotlin.io.use is defined on.
                val dng = DngCreator(c, result)
                try {
                    FileOutputStream(f).use { os -> dng.writeImage(os, img) }
                } finally {
                    dng.close()
                }
                burstFiles.add(f)
                burstStatus?.invoke("Captured ${burstFiles.size}/$burstWanted")
            } catch (t: Throwable) {
                burstStatus?.invoke("Frame dropped: ${t.message}")
            } finally {
                try { img.close() } catch (_: Throwable) {}
            }
        }
        if (burstWanted in 1..burstFiles.size) {
            val done = burstDone
            val out = ArrayList(burstFiles)
            burstDone = null
            burstWanted = 0
            done?.invoke(out, null)
        } else if (burstWanted <= 0 && burstDone != null) {
            // Every frame failed, so nothing will ever satisfy the branch above.
            val done = burstDone
            burstDone = null
            done?.invoke(null, "The camera dropped every frame of the burst")
        }
    }

    /**
     * Hands the preview back to auto after a burst. Without this the viewfinder
     * stays pinned to whatever the shot was metered at, which looks like the
     * camera has frozen when the scene changes.
     */
    fun unlock() {
        val cam = device ?: return
        val s = session ?: return
        val preview = previewSurface ?: return
        try {
            val b = cam.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            b.addTarget(preview)
            b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
            b.set(CaptureRequest.CONTROL_AE_LOCK, false)
            b.set(CaptureRequest.CONTROL_AWB_LOCK, false)
            b.set(CaptureRequest.CONTROL_AF_MODE,
                  CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            b.set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF)
            s.setRepeatingRequest(b.build(), object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(
                    sess: CameraCaptureSession, req: CaptureRequest, result: TotalCaptureResult) {
                    lastResult = result
                }
            }, handler)
        } catch (_: Throwable) {
        }
    }
}
