// JNI surface for the Android build.
//
// Deliberately narrow: DNGs in, a merged DNG and a packed ARGB buffer out. No
// camera. Capturing a RAW burst through Camera2 with fixed exposure and a
// usable ZSL ring is a large piece of work on its own and behaves differently
// on every OEM, so this build processes DNGs the phone (or another camera)
// already produced. That keeps the first Android milestone about whether the
// algorithm runs and what it costs, not about camera fragmentation.
#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/pipeline.h"
#include "core/render_isp.h"
#include "core/dng_writer.h"
#include "core/parallel.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "fuzeframe", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "fuzeframe", __VA_ARGS__)

namespace {

JavaVM* g_vm = nullptr;

std::string jstr(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

// Defaults chosen for a 3GB phone, not for a flagship.
//
// scale 1.0, not 2.0: at 2x the accumulator alone is 22MB per output megapixel,
// so a 13MP sensor would want about 1.1GB for that buffer before anything else.
// At 1x it is roughly 275MB, which is the difference between fitting and not.
//
// Decimate rather than the FFT grey: the FFT runs on the CPU here, and a
// full-resolution complex transform per frame is the single most expensive
// thing this pipeline can be asked to do without a GPU.
hhsr::Config android_config() {
    hhsr::Config cfg;
    cfg.scale = 1.0f;
    cfg.grey_method = hhsr::GreyMethod::Decimate;
    cfg.robustness_save_mask = false;
    return cfg;
}

// Progress can in principle be reported from a worker thread. Today every
// call site is on the thread that entered process(), but a cached JNIEnv used
// from the wrong thread is undefined behaviour that surfaces as an opaque
// abort on a phone, so attach on demand rather than depend on that staying true.
class ProgressBridge {
public:
    ProgressBridge(JNIEnv* env, jobject cb) {
        if (!cb || !g_vm) return;
        obj_ = env->NewGlobalRef(cb);
        jclass cls = env->GetObjectClass(cb);
        method_ = env->GetMethodID(cls, "onProgress", "(Ljava/lang/String;F)V");
        env->DeleteLocalRef(cls);
        if (!method_) {
            env->DeleteGlobalRef(obj_);
            obj_ = nullptr;
        }
    }
    ~ProgressBridge() {
        if (!obj_) return;
        JNIEnv* env = nullptr;
        if (g_vm->GetEnv((void**)&env, JNI_VERSION_1_6) == JNI_OK)
            env->DeleteGlobalRef(obj_);
    }
    ProgressBridge(const ProgressBridge&) = delete;
    ProgressBridge& operator=(const ProgressBridge&) = delete;

    // A no-op when there is no callback, so callers need no null check.
    void report(const std::string& stage, float f) const {
        if (!obj_) return;
        JNIEnv* env = nullptr;
        bool attached = false;
        if (g_vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
            if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return;
            attached = true;
        }
        jstring js = env->NewStringUTF(stage.c_str());
        env->CallVoidMethod(obj_, method_, js, (jfloat)f);
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteLocalRef(js);
        if (attached) g_vm->DetachCurrentThread();
    }

private:
    jobject obj_ = nullptr;
    jmethodID method_ = nullptr;
};

}  // namespace

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    g_vm = vm;
    return JNI_VERSION_1_6;
}

JNIEXPORT jstring JNICALL
Java_ventures_spacetree_fuzeframe_Native_process(
    JNIEnv* env, jclass, jobjectArray dngPaths, jstring outDngPath, jobject cb) {

    const jsize n = dngPaths ? env->GetArrayLength(dngPaths) : 0;
    if (n < 2) return env->NewStringUTF("Pick at least 2 DNG files");

    std::vector<std::string> paths;
    paths.reserve((size_t)n);
    for (jsize i = 0; i < n; ++i) {
        jstring s = (jstring)env->GetObjectArrayElement(dngPaths, i);
        paths.push_back(jstr(env, s));
        env->DeleteLocalRef(s);
    }
    const std::string out = jstr(env, outDngPath);

    // The pipeline reports failures through the progress channel and then just
    // returns an empty preview, so keep the last one: "Processing failed" alone
    // is useless when the answer is "out of memory during merge".
    ProgressBridge bridge(env, cb);
    std::string last_error;
    hhsr::ProgressFn progress = [&bridge, &last_error](const std::string& stage, float f) {
        if (stage.compare(0, 6, "Error:") == 0) last_error = stage;
        LOGI("%s", stage.c_str());
        bridge.report(stage, f);
    };

    try {
        hhsr::Image preview = hhsr::process_burst_paths_to_dng(
            paths, android_config(), out, progress, 512);
        if (preview.h <= 0 || preview.w <= 0)
            return env->NewStringUTF(last_error.empty() ? "Processing failed"
                                                        : last_error.c_str());
    } catch (const std::exception& e) {
        LOGE("process threw: %s", e.what());
        return env->NewStringUTF(e.what());
    } catch (...) {
        return env->NewStringUTF("Processing failed (unknown error)");
    }
    return nullptr;   // null means success
}

// Renders the merged DNG with the same ISP the iOS build uses and returns
// packed ARGB_8888, which Bitmap.createBitmap consumes directly.
//
// ARGB ints rather than RGB bytes on purpose: a byte array would have to be
// widened to ints on the Kotlin side anyway, and on a 13MP frame that
// intermediate is 39MB the phone does not have to spare. Filled a row at a
// time so there is no second full-size native copy either.
JNIEXPORT jintArray JNICALL
Java_ventures_spacetree_fuzeframe_Native_renderArgb(
    JNIEnv* env, jclass, jstring dngPath, jintArray outWH) {

    std::vector<uint16_t> rgb;
    int W = 0, H = 0;
    float wb[3] = {1.f, 1.f, 1.f};
    float m[9] = {1,0,0, 0,1,0, 0,0,1};
    bool has_color = false;
    if (!hhsr::load_linear_dng_rgb16_color(jstr(env, dngPath), rgb, W, H, wb, m, has_color)
        || W <= 0 || H <= 0) {
        LOGE("could not read merged DNG");
        return nullptr;
    }

    hhsr::IspParams p;                       // same defaults as iOS
    hhsr::IspState isp;
    if (!hhsr::isp_analyse(rgb.data(), W, H, has_color ? m : nullptr, p, isp)) {
        LOGE("isp_analyse failed");
        return nullptr;
    }

    jintArray arr = env->NewIntArray((jsize)((size_t)W * (size_t)H));
    if (!arr) {
        LOGE("could not allocate %dx%d pixel array", W, H);
        return nullptr;
    }

    // A band at a time: rows within a band render in parallel across cores, and
    // only the band is held natively. Copying into the Java array happens on
    // this thread, which is the only one holding a valid JNIEnv.
    const int kBand = 64;
    std::vector<jint> band((size_t)kBand * (size_t)W);
    for (int y0 = 0; y0 < H; y0 += kBand) {
        const int bh = std::min(kBand, H - y0);
        hhsr::parallel_rows(bh, 0, [&](int r) {
            const int y = y0 + r;
            jint* dst = band.data() + (size_t)r * (size_t)W;
            const uint16_t* src = rgb.data() + (size_t)y * (size_t)W * 3u;
            for (int x = 0; x < W; ++x) {
                float sr, sg, sb;
                hhsr::isp_render(isp,
                                 src[x * 3 + 0] * (1.f / 65535.f),
                                 src[x * 3 + 1] * (1.f / 65535.f),
                                 src[x * 3 + 2] * (1.f / 65535.f),
                                 x, y, sr, sg, sb);
                // std::lround, matching SRBridge.mm -- not +0.5 truncation, or
                // the Android JPEG would differ from the iOS one by a count in
                // roughly half the pixels.
                const jint rr = (jint)std::lround(sr * 255.f);
                const jint gg = (jint)std::lround(sg * 255.f);
                const jint bb = (jint)std::lround(sb * 255.f);
                dst[x] = (jint)0xFF000000 | (rr << 16) | (gg << 8) | bb;
            }
        });
        env->SetIntArrayRegion(arr, (jsize)((size_t)y0 * (size_t)W),
                               (jsize)((size_t)bh * (size_t)W), band.data());
    }

    if (outWH && env->GetArrayLength(outWH) >= 2) {
        jint wh[2] = {W, H};
        env->SetIntArrayRegion(outWH, 0, 2, wh);
    }
    return arr;
}

}  // extern "C"
