// srsr_pipeline.cpp -- whole-burst SRSR (GCam 6.1 SABRE port, Documents/srsr)
// orchestration: geometry, the Monte-Carlo noise model, the per-frame GPU
// analysis, and the banded merge + denoise streamed into the DNG writer.
//
// Memory plan (the run is sized to Config::srsr_memory_budget_mb, 2.3 GB):
//   * Frames go through TWO staging buffers; a decoded frame is copied in, its
//     PREPARE turns it into a 2-byte/px quad texture, and the buffer is reused.
//     The host never holds more than the frame being decoded and the one being
//     uploaded.
//   * Resident per frame through the merge: quad codes (RGBA16Uint), kernel
//     covariance (RGBA16Float), the dilated mask (R8) and the flow -- ~51 MB per
//     12 MP frame. Alignment pyramids, guides and staging are freed before the
//     output stage.
//   * MERGE and DENOISE run on horizontal output bands with the denoiser's halo,
//     and the last level writes uint16 DNG rows into a shared buffer the writer
//     consumes directly. The full-resolution output never exists anywhere, at
//     1x or 2x. Two output buffers let the CPU encode band i-1 while the GPU
//     computes band i.
// Speed: the next frame is decoded on a worker thread while the GPU analyses
// the current one; everything per pixel runs on the GPU.
#include "srsr_pipeline.h"
#include "srsr_backend.h"
#include "dng_writer.h"
#include "pipeline.h"
#include "raw_io.h"
#include "parallel.h"
#include "prof.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <map>
#include <mutex>
#include <vector>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace hhsr {

// pipeline_paths.cpp: the preview sampler the normal path uses.
void srsr_preview_store_sample(Image& preview, int py, int px, f32 cn0, f32 cn1, f32 cn2,
                               const Config& work, int nch);

namespace srsr {
namespace {

// ---------------------------------------------------------------- tuning
// SABRE 6.1 constants (Documents/srsr SRSRTuning, pinned to .rodata there).
struct Tuning {
    float greenDifferenceImpact = 10.0f, shrinkedSigma = 8.0f, stretchedSigma = 1.0f,
          gradientClip = 0.001f, baseSigma = 4.0f, blurredSigma = 1.0f, gradientTransition = 142.857f;
    float distanceMultiplier = 1.0f, rejectionThreshold = 0.0f, extraMotionRobustnessBoost = 6.0f,
          motionRobustnessBoostStdDevThreshold = 5.0f, motionThresholdAtReferenceWidth = 1.0e-4f,
          unblockerThresholdAtReferenceWidth = 9.999e-5f, geometryReferenceWidth = 2016.0f;
    float unblockerVarianceScale = 0.2f, unblockerBias = 0.05f;
    float coverageThreshold = 4.0f, covarianceMultiplier = 0.3f, kernelEpsilon = 0.00005f,
          footprintLimit = 1.5f, normalization = 1.0f, accumulatedWeightScale = 25.0f,
          mergeBorderPaddingPixels = 2.5f, dehomogenizeEpsilon = 1.0e-7f;
    float weightTextureScale = 25.0f, weightTextureBias = 1.0f, denoisingFactor = 0.0625f,
          outlierSigmaMultiplier = 0.125f, rejectLumaRevertFactor = 0.4f, globalLumaRevertFactor = 1.0f,
          vignetteStrengthBoost = 0.0f, vignetteSpatialScaleFactor = 1.0f;
};

// Halide-reference alignment configuration (Downloads/alignment): finest first.
static const int kScaleFactors[] = {2, 4, 4};
static const int kTileSizes[] = {32, 32, 16, 8};
static const int kSearchRadii[] = {1, 3, 4, 4};
static const bool kNormL2[] = {false, true, true, true};
static const int kIcaIterations = 3;
static const int kOutputTile = 16, kOutputStride = 8;
static const int kLevels = 4;

// ---------------------------------------------------------------- geometry
struct LevelGeo { int w, h, tile, radius; bool l2; int tx() const { return w / tile; } int ty() const { return h / tile; } };
struct TransferGeo { int targetTile, targetStride, targetTilesX, targetTilesY; float gridScaleRatio, flowScale, offsetPx; };

struct Geometry {
    int rawW = 0, rawH = 0, cropX = 0, cropY = 0;
    int quadW = 0, quadH = 0, bayerW = 0, bayerH = 0, outW = 0, outH = 0;
    std::vector<LevelGeo> levels;
    std::vector<TransferGeo> transfers;           // coarse -> fine
    std::vector<std::pair<int, int>> chain;       // every 2x halving from the quad grid
    std::vector<int> levelChain;                  // chain index of each level
    int flowTX = 0, flowTY = 0;
    TransferGeo finalTransfer{};
    float flowScaleX = 1, flowScaleY = 1, flowOffsetX = 0, flowOffsetY = 0;
    int unbW = 1, unbH = 1;
    float unbScaleX = 1, unbScaleY = 1;
    float motionThreshold = 0, unblockerReductionThreshold = 0;
    float borderMinX = 0, borderMinY = 0, borderMaxX = 1, borderMaxY = 1;

    // SRSRGeometry.swift, with the output grid decoupled from the Bayer grid.
    bool build(int rw, int rh, const CFA& cfa, float scale, const Tuning& t, std::string& err) {
        rawW = rw; rawH = rh;
        // Crop phase so the first quad reads R G1 / G2 B (Documents/srsr D1).
        int ri = -1, rj = -1;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                if (cfa.p[i][j] == 0 && ri < 0) { ri = i; rj = j; }
        if (ri < 0) { err = "CFA has no red site"; return false; }
        // A Bayer pattern is RGGB at that phase: check the other three sites.
        const int gx = cfa.p[ri][1 - rj], gy = cfa.p[1 - ri][rj], bb = cfa.p[1 - ri][1 - rj];
        if (gx != 1 || gy != 1 || bb != 2) { err = "CFA is not a Bayer pattern"; return false; }
        cropX = rj; cropY = ri;
        quadW = (rawW - cropX) / 2;
        quadH = (rawH - cropY) / 2;
        bayerW = 2 * quadW; bayerH = 2 * quadH;
        const float s = (scale > 0.f && std::isfinite(scale)) ? scale : 1.f;
        outW = std::max(1, (int)std::lround(s * bayerW));
        outH = std::max(1, (int)std::lround(s * bayerH));

        chain.assign(1, {quadW, quadH});
        levelChain.assign(1, 0);
        std::vector<std::pair<int, int>> shapes(1, {quadW, quadH});
        for (int f : kScaleFactors) {
            int w = shapes.back().first, h = shapes.back().second;
            for (int r = f; r > 1; r >>= 1) {
                w = w / 2 + w % 2;
                h = h / 2 + h % 2;
                chain.push_back({w, h});
            }
            levelChain.push_back((int)chain.size() - 1);
            shapes.push_back({w, h});
        }
        levels.clear();
        for (int i = 0; i < kLevels; ++i) {
            const LevelGeo lv{shapes[i].first, shapes[i].second, kTileSizes[i], kSearchRadii[i], kNormL2[i]};
            if (lv.w < lv.tile || lv.h < lv.tile) { err = "image too small for the alignment pyramid"; return false; }
            levels.push_back(lv);
        }
        const int coarsest = kLevels - 1;
        transfers.clear();
        int flowTile = levels[coarsest].tile, flowStride = levels[coarsest].tile;
        for (int level = coarsest; level > 0; --level) {
            const int target = level - 1;
            const int tt = levels[target].tile, ts = tt;
            const float px = (float)kScaleFactors[target];
            TransferGeo tg;
            tg.targetTile = tt; tg.targetStride = ts;
            tg.targetTilesX = (levels[target].w - tt) / ts + 1;
            tg.targetTilesY = (levels[target].h - tt) / ts + 1;
            tg.gridScaleRatio = px * (float)flowStride / (float)ts;
            tg.flowScale = px;
            tg.offsetPx = (float)flowTile * px / 2.0f - (float)tt / 2.0f;
            transfers.push_back(tg);
            flowTile = tt; flowStride = ts;
        }
        flowTX = (quadW - kOutputTile) / kOutputStride + 1;
        flowTY = (quadH - kOutputTile) / kOutputStride + 1;
        if (flowTX <= 0 || flowTY <= 0) { err = "image too small for the flow grid"; return false; }
        finalTransfer = TransferGeo{kOutputTile, kOutputStride, flowTX, flowTY,
                                    (float)flowStride / (float)kOutputStride, 1.0f,
                                    (float)flowTile / 2.0f - (float)kOutputTile / 2.0f};
        flowScaleX = (float)quadW / (float)(kOutputStride * flowTX);
        flowScaleY = (float)quadH / (float)(kOutputStride * flowTY);
        flowOffsetX = -0.5f / (float)flowTX;
        flowOffsetY = -0.5f / (float)flowTY;
        unbW = ((bayerW + 7) & ~7) / 8;
        unbH = ((bayerH + 7) & ~7) / 8;
        unbScaleX = (float)bayerW / (float)(8 * unbW);
        unbScaleY = (float)bayerH / (float)(8 * unbH);
        const float widthRatio = t.geometryReferenceWidth / (float)quadW;
        motionThreshold = widthRatio * t.motionThresholdAtReferenceWidth;
        unblockerReductionThreshold = widthRatio * t.unblockerThresholdAtReferenceWidth;
        // The 4x4 footprint reaches 2.5 BAYER pixels from the sample: the border is
        // in Bayer units at any output scale (Documents/srsr had outWidth, = Bayer at 1x).
        borderMinX = t.mergeBorderPaddingPixels / (float)bayerW;
        borderMinY = t.mergeBorderPaddingPixels / (float)bayerH;
        borderMaxX = 1.0f - borderMinX;
        borderMaxY = 1.0f - borderMinY;
        return true;
    }

    size_t frame_bytes() const {     // resident per frame through the merge
        const size_t q = (size_t)quadW * quadH;
        return q * 8 + q * 8 + q + (size_t)flowTX * flowTY * 16;
    }
    size_t analysis_bytes() const {  // freed before the output stage
        const size_t q = (size_t)quadW * quadH;
        size_t s = q * 8 * 2 + q + q * 4 + (size_t)unbW * unbH;
        for (auto& c : chain) s += (size_t)c.first * c.second * 4 * 2;
        for (auto& l : levels) s += (size_t)l.w * l.h * 4 * 2 + (size_t)l.tx() * l.ty() * 16 * 4;
        s += (size_t)flowTX * flowTY * 16 * 2;
        s += (size_t)rawW * rawH * 4 * 2;                 // staging
        return s;
    }
};

// ---------------------------------------------------------------- noise
// Platform-stable RNG (PCG32 + Box-Muller): the same LUT on the PC and the phone.
struct Rng {
    uint64_t s;
    bool has = false;
    float spare = 0.f;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    uint32_t next() {
        const uint64_t old = s;
        s = old * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint32_t xs = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        const uint32_t rot = (uint32_t)(old >> 59u);
        return (xs >> rot) | (xs << ((32u - rot) & 31u));
    }
    float uni() { return ((float)(next() >> 8) + 0.5f) * (1.0f / 16777216.0f); }   // (0, 1)
    float normal() {
        if (has) { has = false; return spare; }
        const float u1 = uni(), u2 = uni();
        const float r = std::sqrt(-2.0f * std::log(u1));
        const float th = 6.28318530718f * u2;
        spare = r * std::sin(th); has = true;
        return r * std::cos(th);
    }
};

static inline float half_round(float v) {
    if (!(v == v)) return v;
    const float a = std::fabs(v);
    if (a >= 65520.0f) return std::copysign(65504.0f, v);
    if (a < 6.103515625e-05f) return std::copysign(std::rint(a * 16777216.0f) / 16777216.0f, v);
    int e;
    std::frexp(a, &e);
    const float step = std::ldexp(1.0f, e - 11);
    return std::copysign(std::rint(a / step) * step, v);
}

// One P x P patch of a flat, neutral (equal S2 on every site) scene at S2 level s,
// with frame `n`'s noise, through exactly what the loader and PREPARE do:
// sensor noise, the sensor's [0, 1] clip, pre-whitening, the exposure-match gain
// with its saturation cap (apply_raw_gain), the uint16 quad code. Output: the
// SQRT of each site's S2 (the guide's perceptual domain), 4 per quad.
static void simulate_patch(const FrameNoise& n, float s, int P, Rng& rng, std::vector<float>& sq) {
    static const int col[4] = {0, 1, 1, 2};
    sq.resize((size_t)P * P * 4);
    for (size_t i = 0; i < (size_t)P * P; ++i)
        for (int k = 0; k < 4; ++k) {
            const int c = col[k];
            const float w = n.wb[c], g = n.gain;
            float x0 = s / std::max(1e-6f, g * w);
            x0 = std::min(x0, 1.0f);
            const float var = std::max(0.0f, n.alpha[c] * x0 + n.beta[c]);
            float x = x0 + std::sqrt(var) * rng.normal();
            x = std::min(std::max(x, 0.0f), 1.0f);
            const float v = x * w;
            float q = v * g;
            if (g > 1.f && q > w) q = std::max(v, w);
            const float code = std::min(65535.0f, std::floor(q * kSRSRQuadScale + 0.5f));
            sq[i * 4 + k] = std::sqrt(code / kSRSRQuadScale);
        }
}

struct GuideSample { float w; float gd; float tensor; };

// srsr_guide at quad (x, y) of a sqrt patch: the stored std_dev (half-rounded like
// the RGBA16Float texture), the unshrunk green difference, and the structure
// tensor's noise power per direction (half its trace, rotation-invariant).
static void mc_guide(const std::vector<float>& sq, int P, int x, int y, GuideSample& o) {
    auto WG = [](int d) { return d == 0 ? 0.5f : 0.25f; };
    float m1[3] = {0, 0, 0}, m0[3] = {0, 0, 0}, sums[9];
    float gd = 0.f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = std::min(P - 1, std::max(0, x + dx)), yy = std::min(P - 1, std::max(0, y + dy));
            const float* q = &sq[((size_t)yy * P + xx) * 4];
            const float rgb[3] = {q[0], q[1] + q[2], q[3]};
            sums[(dy + 1) * 3 + dx + 1] = rgb[0] * 0.25f + rgb[1] * 0.25f + rgb[2] * 0.25f;
            for (int c = 0; c < 3; ++c) { m0[c] += rgb[c]; m1[c] += rgb[c] * rgb[c]; }
            gd += std::fabs(q[1] - q[2]) * WG(dx) * WG(dy);
        }
    float var = 0.f;
    for (int c = 0; c < 3; ++c) { m0[c] /= 9.f; m1[c] /= 9.f; var += (m1[c] - m0[c] * m0[c]) * (1.f / 9.f); }
    o.w = half_round(std::max(std::sqrt(std::max(var, 0.f)), gd));
    o.gd = gd;
    float dxx = 0, dyy = 0;
    for (int yy = 0; yy < 2; ++yy)
        for (int xx = 0; xx < 2; ++xx) {
            const float ddx = sums[(yy + 1) * 3 + (xx + 1)] - sums[yy * 3 + xx];
            const float ddy = sums[yy * 3 + (xx + 1)] - sums[(yy + 1) * 3 + xx];
            dxx += ddx * ddx; dyy += ddy * ddy;
        }
    o.tensor = 0.5f * (dxx + dyy) / 8.f;
}

} // namespace

FrameNoise frame_noise(const Config& cfg, float gain) {
    FrameNoise n;
    for (int c = 0; c < 3; ++c) {
        n.alpha[c] = std::isfinite(cfg.alpha_dng[c]) ? std::max(0.f, cfg.alpha_dng[c]) : 0.f;
        n.beta[c] = std::isfinite(cfg.beta_dng[c]) ? std::max(0.f, cfg.beta_dng[c]) : 0.f;
        float w = 1.f;
        if (cfg.raw_prewhitened) {
            const float wg = cfg.white_balance[1], wc = cfg.white_balance[c];
            if (std::isfinite(wg) && std::isfinite(wc) && wg > 0.f && wc > 0.f) w = wc / wg;
        }
        n.wb[c] = w;
    }
    n.gain = (gain > 0.f && std::isfinite(gain)) ? gain : 1.f;
    return n;
}

// Monte Carlo, GCam semantics: the LUT holds PER-SAMPLE noise in the guide's colour
// space (one quad, sqrt domain), the way GCam builds it -- the 8.x shader then
// applies kFilterVarianceScale itself, 6.1 consumes it as is, and the 5x5 dilate's
// tolerance is calibrated for exactly that. (The noise power of the filtered,
// bicubic-sampled guide difference is ~0.09x of this, = GCam's kFilterVarianceScale
// 0.0899; used as the Wiener term it leaves ~0.07 of rejection per pixel on pure
// noise, which the 25-tap /3 dilate turns into ~50% lost weight in flat areas.)
// Each statistic is measured on a simulated flat, neutral patch at luma i/9 pushed
// through the frame's real chain: sensor noise (DNG NoiseProfile), the sensor clip,
// pre-whitening, the exposure-match gain with its saturation cap, the quad code,
// the sqrt.
//   row0.xyz  Var of one quad's (Y, Co, Cg), mean of the base and current frame
//             (the shader doubles it: Var(a - b) = Var(a) + Var(b))
//   row0.w    E[guide std_dev] under noise, mean of the two frames (the floor)
//   row1.x    E[green_differences] of the current frame (its own guide's Wiener)
//   row1.y    E[structure-tensor noise power per direction] of the current frame
void build_noise_lut(const FrameNoise& base, const FrameNoise& cur, float impact, SRSRNoiseLUT& out) {
    struct Key { float v[22]; bool operator<(const Key& o) const { return std::memcmp(v, o.v, sizeof v) < 0; } };
    Key key;
    const FrameNoise* fs[2] = {&base, &cur};
    for (int f = 0; f < 2; ++f) {
        for (int c = 0; c < 3; ++c) {
            key.v[f * 10 + c] = fs[f]->alpha[c];
            key.v[f * 10 + 3 + c] = fs[f]->beta[c];
            key.v[f * 10 + 6 + c] = fs[f]->wb[c];
        }
        key.v[f * 10 + 9] = fs[f]->gain;
    }
    key.v[20] = impact; key.v[21] = 1.f;    // [21]: LUT semantics version
    static std::mutex mu;
    static std::map<Key, SRSRNoiseLUT> cache;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(key);
        if (it != cache.end()) { out = it->second; return; }
    }
    constexpr int P = 48, R = 3;
    SRSRNoiseLUT lut;
    std::memset(&lut, 0, sizeof lut);
    parallel_rows(kSRSRNoiseLUTColumns, 0, [&](int i) {
        const float p = (float)i / (float)(kSRSRNoiseLUTColumns - 1);
        const float s = p * p;
        Rng rng(0x5EED0000ULL + (uint64_t)i * 7919ULL);
        std::vector<float> patch;
        double m[2][3] = {{0, 0, 0}, {0, 0, 0}}, m2[2][3] = {{0, 0, 0}, {0, 0, 0}}, sw[2] = {0, 0};
        double sgd = 0, sten = 0;
        long nq = 0, ng = 0;
        GuideSample gs;
        for (int r = 0; r < R; ++r) {
            for (int f = 0; f < 2; ++f) {
                simulate_patch(*fs[f], s, P, rng, patch);
                for (size_t q = 0; q < (size_t)P * P; ++q) {
                    const float* v = &patch[q * 4];
                    const float rr = v[0], gg = v[1] + v[2], bb = v[3];
                    const float ycc[3] = {0.25f * (rr + gg + bb), 0.25f * (-rr + gg - bb), 0.5f * (rr - bb)};
                    for (int c = 0; c < 3; ++c) { m[f][c] += ycc[c]; m2[f][c] += (double)ycc[c] * ycc[c]; }
                }
                for (int y = 1; y < P - 1; ++y)
                    for (int x = 1; x < P - 1; ++x) {
                        mc_guide(patch, P, x, y, gs);
                        sw[f] += gs.w;
                        if (f == 1) { sgd += gs.gd; sten += gs.tensor; }
                    }
            }
            nq += (long)P * P;
            ng += (long)(P - 2) * (P - 2);
        }
        for (int c = 0; c < 3; ++c) {
            double v = 0;
            for (int f = 0; f < 2; ++f) {
                const double mu_ = m[f][c] / nq;
                v += 0.5 * std::max(0.0, m2[f][c] / nq - mu_ * mu_);
            }
            lut.row0[i][c] = std::max(1e-12f, (float)v);
        }
        lut.row0[i][3] = std::max(1e-6f, (float)(0.5 * (sw[0] + sw[1]) / ng));
        lut.row1[i][0] = std::max(1e-12f, (float)(sgd / ng));
        lut.row1[i][1] = std::max(1e-12f, (float)(sten / ng));
    });
    (void)impact;
    {
        std::lock_guard<std::mutex> lk(mu);
        if (cache.size() > 64) cache.clear();
        cache[key] = lut;
    }
    out = lut;
}

void yuv_noise_model(const FrameNoise& b, float out[4]) {
    // Per-colour S2 variance of one frame: Var(s) = (g w a) s + (g w)^2 b.
    float n[3], f[3];
    for (int c = 0; c < 3; ++c) {
        const float gw = b.gain * b.wb[c];
        n[c] = gw * b.alpha[c];
        f[c] = gw * gw * b.beta[c];
    }
    const float c0 = 0.045215346f, c1 = 0.51146638f, c2 = 0.0052117277f;   // squared Rec.709
    out[0] = n[0] * c0 + n[1] * c1 + n[2] * c2;
    out[1] = -n[0] * c0 + n[1] * c1 * 0.297327f;
    out[2] = n[1] * c1 * 0.100944f - n[2] * c2;
    out[3] = f[0] * c0 + f[1] * c1 + f[2] * c2;
}

namespace {

static inline size_t r16(size_t b) { return (b + 15) & ~(size_t)15; }

SRSRFrameNormalization quad_norm() {
    SRSRFrameNormalization n;
    const float g = 1.0f / kSRSRQuadScale;
    n.gain0 = n.gain1 = n.gain2 = n.gain3 = g;
    n.blackTimesGain0 = n.blackTimesGain1 = n.blackTimesGain2 = n.blackTimesGain3 = 0.f;
    return n;
}

struct Resources {
    // per slot
    std::vector<std::unique_ptr<Tex>> quad, cov, flow, dilated;
    // analysis
    std::unique_ptr<Tex> guideBase, guideScratch, rejection, dilatePartial, unblocker;
    std::vector<std::unique_ptr<Tex>> refChain, movChain, gx, gy, hess, flowIn, flowA, flowB;
    std::unique_ptr<Tex> flowFinal, flowMedian;
    std::unique_ptr<Buf> staging[2];
    // output
    std::unique_ptr<Tex> accW, dummyR8, dummyFlow;
    std::unique_ptr<Tex> bandA, bandB;
    std::unique_ptr<Buf> out[2];
    std::unique_ptr<Buf> maskOut;      // Config::robustness_save_mask
    void free_analysis() {
        guideBase.reset(); guideScratch.reset(); rejection.reset(); dilatePartial.reset(); unblocker.reset();
        refChain.clear(); movChain.clear(); gx.clear(); gy.clear(); hess.clear();
        flowIn.clear(); flowA.clear(); flowB.clear();
        flowFinal.reset(); flowMedian.reset();
        staging[0].reset(); staging[1].reset();
    }
};

struct Run {
    Backend& be;
    const Geometry& G;
    const Tuning& T;
    const Config& work;
    Resources& R;

    void prepare(int slot, Buf* src, Tex* grey) {
        SRSRPrepareUniforms u{};
        u.quadWidth = G.quadW; u.quadHeight = G.quadH;
        u.cropOriginX = G.cropX; u.cropOriginY = G.cropY;
        u.rawWidth = G.rawW; u.rawHeight = G.rawH;
        u.perm0 = 0; u.perm1 = 1; u.perm2 = 2; u.perm3 = 3;
        u.quadScale = kSRSRQuadScale;
        u.norm = quad_norm();
        be.buf(1, src);
        be.tex(1, R.quad[slot].get());
        be.tex(2, grey);
        be.bytes(0, &u, sizeof u);
        be.dispatch(K::prepare, G.quadW, G.quadH);
    }

    void downsample_chain(std::vector<std::unique_ptr<Tex>>& chain) {
        for (size_t i = 1; i < chain.size(); ++i) {
            SRSRDownsampleUniforms u{chain[i - 1]->w, chain[i - 1]->h, chain[i]->w, chain[i]->h};
            be.tex(0, chain[i - 1].get());
            be.tex(1, chain[i].get());
            be.bytes(0, &u, sizeof u);
            be.dispatch(K::downsample, chain[i]->w, chain[i]->h);
        }
    }

    void reference_gradients() {
        for (int l = 0; l < kLevels; ++l) {
            const LevelGeo& lv = G.levels[l];
            SRSRGradientUniforms u{lv.w, lv.h, lv.tile, lv.tx(), lv.ty()};
            be.tex(0, R.refChain[G.levelChain[l]].get());
            be.tex(1, R.gx[l].get());
            be.tex(2, R.gy[l].get());
            be.bytes(0, &u, sizeof u);
            be.dispatch(K::gradients, lv.w, lv.h);
            be.tex(0, R.gx[l].get());
            be.tex(1, R.gy[l].get());
            be.tex(2, R.hess[l].get());
            be.bytes(0, &u, sizeof u);
            be.dispatch(K::gradient_tile_sums, lv.tx(), lv.ty());
        }
    }

    void flow_transfer(const TransferGeo& t, int imageLevel, int srcTX, int srcTY, Tex* src, Tex* dst) {
        const LevelGeo& im = G.levels[imageLevel];
        SRSRFlowTransferUniforms u{};
        u.width = im.w; u.height = im.h;
        u.sourceTilesX = srcTX; u.sourceTilesY = srcTY;
        u.targetTileSize = t.targetTile; u.targetStride = t.targetStride;
        u.targetTilesX = t.targetTilesX; u.targetTilesY = t.targetTilesY;
        u.gridScaleRatio = t.gridScaleRatio; u.flowScale = t.flowScale;
        u.offsetXPx = t.offsetPx; u.offsetYPx = t.offsetPx;
        be.tex(0, src);
        be.tex(1, R.refChain[G.levelChain[imageLevel]].get());
        be.tex(2, R.movChain[G.levelChain[imageLevel]].get());
        be.tex(3, dst);
        be.bytes(0, &u, sizeof u);
        be.dispatch(K::flow_transfer, t.targetTilesX, t.targetTilesY);
    }

    void align(int slot) {
        const int coarsest = kLevels - 1;
        {
            SRSRFlowMedianUniforms u{G.levels[coarsest].tx(), G.levels[coarsest].ty()};
            be.tex(0, R.flowIn[coarsest].get());
            be.bytes(0, &u, sizeof u);
            be.dispatch(K::clear_flow, u.tilesX, u.tilesY);
        }
        Tex* cur = R.flowIn[coarsest].get();
        for (int level = coarsest; level >= 0; --level) {
            const LevelGeo& lv = G.levels[level];
            Tex* ref = R.refChain[G.levelChain[level]].get();
            Tex* mov = R.movChain[G.levelChain[level]].get();
            {
                const int span = 2 * lv.radius + 1, cand = span * span;
                SRSRBlockMatchingUniforms u{lv.w, lv.h, lv.tile, lv.tx(), lv.ty(), lv.radius, cand, lv.l2 ? 1 : 0};
                be.tex(0, ref); be.tex(1, mov); be.tex(2, cur); be.tex(3, R.flowA[level].get());
                be.bytes(0, &u, sizeof u);
                be.dispatch_bm(lv.tx(), lv.ty(), cand, r16((size_t)cand * 4), r16((size_t)lv.tile * lv.tile * 4));
                cur = R.flowA[level].get();
            }
            Tex* srcT = R.flowA[level].get();
            Tex* dstT = R.flowB[level].get();
            for (int it = 0; it < kIcaIterations; ++it) {
                SRSRICAUniforms u{lv.w, lv.h, lv.tile, lv.tx(), lv.ty()};
                be.tex(0, ref); be.tex(1, mov);
                be.tex(2, R.gx[level].get()); be.tex(3, R.gy[level].get()); be.tex(4, R.hess[level].get());
                be.tex(5, srcT); be.tex(6, dstT);
                be.bytes(0, &u, sizeof u);
                be.dispatch(K::ica_iteration, lv.tx(), lv.ty());
                std::swap(srcT, dstT);
                cur = srcT;    // the last write
            }
            if (level == 0) break;
            flow_transfer(G.transfers[coarsest - level], level - 1, lv.tx(), lv.ty(), cur, R.flowIn[level - 1].get());
            cur = R.flowIn[level - 1].get();
        }
        flow_transfer(G.finalTransfer, 0, G.levels[0].tx(), G.levels[0].ty(), cur, R.flowFinal.get());
        {
            SRSRFlowMedianUniforms u{G.flowTX, G.flowTY};
            be.tex(0, R.flowFinal.get()); be.tex(1, R.flowMedian.get());
            be.bytes(0, &u, sizeof u);
            be.dispatch(K::flow_median, G.flowTX, G.flowTY);
        }
        {
            SRSRFlowPackUniforms u{G.flowTX, G.flowTY, 1.0f / (float)G.quadW, 1.0f / (float)G.quadH};
            be.tex(0, R.flowMedian.get()); be.tex(1, R.flow[slot].get());
            be.bytes(0, &u, sizeof u);
            be.dispatch(K::flow_pack, G.flowTX, G.flowTY);
        }
    }

    void guide(int slot, Tex* guideTarget, const SRSRNoiseLUT& lut) {
        SRSRGuideUniforms u{};
        u.quadWidth = G.quadW; u.quadHeight = G.quadH;
        u.greenDifferenceImpact = T.greenDifferenceImpact;
        u.shrinkedSigma = T.shrinkedSigma; u.stretchedSigma = T.stretchedSigma;
        u.gradientClip = T.gradientClip; u.baseSigma = T.baseSigma;
        u.blurredSigma = T.blurredSigma; u.gradientTransition = T.gradientTransition;
        u.norm = quad_norm();
        be.tex(0, R.quad[slot].get()); be.tex(1, guideTarget); be.tex(2, R.cov[slot].get());
        be.bytes(0, &u, sizeof u);
        be.bytes(1, &lut, sizeof lut);
        be.dispatch(K::guide, G.quadW, G.quadH);
    }

    SRSRRejectionUniforms rejection_uniforms() const {
        SRSRRejectionUniforms u{};
        u.quadWidth = G.quadW; u.quadHeight = G.quadH;
        u.flowTilesX = G.flowTX; u.flowTilesY = G.flowTY;
        u.unblockerWidth = G.unbW; u.unblockerHeight = G.unbH;
        u.flowScaleX = G.flowScaleX; u.flowScaleY = G.flowScaleY;
        u.flowOffsetX = G.flowOffsetX; u.flowOffsetY = G.flowOffsetY;
        u.unblockerScaleX = G.unbScaleX; u.unblockerScaleY = G.unbScaleY;
        u.distanceMultiplier = T.distanceMultiplier; u.rejectionThreshold = T.rejectionThreshold;
        u.extraMotionRobustnessBoost = T.extraMotionRobustnessBoost;
        u.motionRobustnessBoostStdDevThreshold = T.motionRobustnessBoostStdDevThreshold;
        u.extraMotionRobustnessMotionThreshold = G.motionThreshold;
        u.unblockerReductionThreshold = G.unblockerReductionThreshold;
        u.unblockerEnabled = work.srsr_unblocker ? 1 : 0;
        return u;
    }

    void rejection_and_dilate(int slot, const SRSRNoiseLUT& lut) {
        const SRSRRejectionUniforms u = rejection_uniforms();
        be.tex(0, R.guideBase.get()); be.tex(1, R.guideScratch.get()); be.tex(2, R.flow[slot].get());
        be.tex(3, R.unblocker.get()); be.tex(4, R.rejection.get());
        be.bytes(0, &u, sizeof u);
        be.bytes(1, &lut, sizeof lut);
        be.dispatch(K::rejection, G.quadW, G.quadH);
        be.tex(0, R.rejection.get()); be.tex(1, R.dilatePartial.get());
        be.bytes(0, &u, sizeof u);
        be.dispatch(K::dilate_h, G.quadW, G.quadH);
        be.tex(0, R.dilatePartial.get()); be.tex(1, R.dilated[slot].get());
        be.bytes(0, &u, sizeof u);
        be.dispatch(K::dilate_v, G.quadW, G.quadH);
    }

    void unblocker_map() {
        SRSRUnblockerUniforms u{};
        u.quadWidth = G.quadW; u.quadHeight = G.quadH; u.outWidth = G.unbW; u.outHeight = G.unbH;
        u.varianceScale = T.unblockerVarianceScale; u.bias = T.unblockerBias;
        u.maxWhite = 1.0e30f;    // the loader's values are already clipped at saturation
        u.norm = quad_norm();
        be.tex(0, R.quad[0].get()); be.tex(1, R.unblocker.get());
        be.bytes(0, &u, sizeof u);
        be.dispatch(K::unblocker, G.unbW, G.unbH);
    }
};

static bool mk(std::unique_ptr<Tex>& t, Backend& be, Fmt f, int w, int h, const char* label) {
    t = be.make_tex(f, std::max(1, w), std::max(1, h), label);
    return t != nullptr;
}

} // namespace
} // namespace srsr

Image process_burst_srsr_to_dng(int frame_count, const RawFrameLoaderFn& loader,
                                const Config& cfg, const std::string& dng_path,
                                const ProgressFn& progress, int maxPreviewDim,
                                std::string* why) {
    using namespace srsr;
    auto report = [&](const std::string& s, float f) { if (progress) progress(s, f); };
    auto fail = [&](const std::string& m) -> Image {
        if (why) *why = m;
        return Image();
    };
    if (frame_count < 1 || !loader) return fail("no frames");

    prof_reset();
    prof_mark_memory("srsr:begin");
    const double t_burst = prof_now_ms();
    std::unique_ptr<Backend> be = make_backend();
    if (!be) return fail("no SRSR GPU backend");

    Tuning T;
    Config work = cfg;
    work.burst_frame_count = frame_count;

    // ---- reference (frame 0: the shutter frame of a capture, the first file of an import)
    report("SRSR: loading reference frame", 0.02f);
    double t0 = prof_now_ms();
    Image ref = loader(0, work, true, 0, 0);
    prof_add_cpu("srsr:decode", prof_now_ms() - t0);
    if (ref.h <= 0 || ref.w <= 0 || ref.c != 1 || ref.data.size() != (size_t)ref.h * ref.w)
        return fail("reference frame could not be decoded");
    const float exp_ref = ref.exposure;
    // Exposure match exactly as the normal loader wrapper (process_burst_loader_to_dng_impl):
    // one gain = reference / frame exposure, brightened pixels capped at each site's saturation.
    float sat[4];
    {
        float med[4], mx[4];
        bool need_stats = false;
        const bool wb_ok = std::isfinite(work.white_balance[1]) && work.white_balance[1] > 0.f;
        for (int s = 0; s < 4; ++s) {
            int col = (int)work.cfa.p[s / 2][s % 2];
            col = std::min(2, std::max(0, col));
            const bool ok = wb_ok && std::isfinite(work.white_balance[col]) && work.white_balance[col] > 0.f;
            sat[s] = ok ? work.white_balance[col] / work.white_balance[1] : -1.f;
            need_stats = need_stats || !ok;
        }
        if (need_stats) {
            raw_site_stats(ref, med, mx);
            for (int s = 0; s < 4; ++s) if (sat[s] < 0.f) sat[s] = mx[s];
        }
    }

    Geometry G;
    std::string gerr;
    if (!G.build(ref.w, ref.h, work.cfa, work.scale, T, gerr)) return fail(gerr);

    // ---- frames and the memory budget
    int n = std::min({frame_count, (int)kSRSRMaxFrames, std::max(2, work.srsr_max_frames)});
    const int dil_base = std::max(1, (int)std::lround(work.scale));
    const int halo = work.srsr_denoise ? 7 * dil_base : 0;       // 1 + 2 + 4 levels
    const size_t row_bytes = (size_t)G.outW * (2 * 8 + 2 * 6);    // 2 band textures + 2 out buffers
    const double budget = (double)std::max(800.f, work.srsr_memory_budget_mb) * 1024.0 * 1024.0;
    const double base_fp = (double)prof_footprint_bytes();        // 0 where unavailable
    const double host_frames = 3.0 * (double)ref.w * ref.h * 4;   // ref + decode in flight + upload
    int band = 1024;
    auto peak_for = [&](int nf, int b) {
        const double resident = (double)nf * G.frame_bytes() + (double)G.quadW * G.quadH *
                                (work.robustness_save_mask ? 2.0 : 1.0);
        const double analysis = resident + (double)G.analysis_bytes() + host_frames;
        const double output = resident + (double)(b + 2 * halo) * row_bytes + 64.0 * 1024 * 1024;  // + writer
        return base_fp + std::max(analysis, output);
    };
    while (band > 128 && peak_for(n, band) > budget) band /= 2;
    while (n > 2 && peak_for(n, band) > budget) --n;
    if (peak_for(n, band) > budget)
        return fail("not enough memory for SRSR at this resolution");
    const double planned_mb = peak_for(n, band) / (1024.0 * 1024.0);
    if (n < frame_count)
        report("SRSR: merging " + std::to_string(n) + " of " + std::to_string(frame_count) + " frames", 0.03f);

    // ---- resources
    Resources R;
    R.quad.resize(n); R.cov.resize(n); R.flow.resize(n); R.dilated.resize(n);
    bool ok = true;
    for (int s = 0; s < n && ok; ++s) {
        ok = ok && mk(R.quad[s], *be, Fmt::RGBA16Uint, G.quadW, G.quadH, "srsr.quad");
        ok = ok && mk(R.cov[s], *be, Fmt::RGBA16Float, G.quadW, G.quadH, "srsr.cov");
        if (s > 0) {
            ok = ok && mk(R.flow[s], *be, Fmt::RGBA32Float, G.flowTX, G.flowTY, "srsr.flow");
            ok = ok && mk(R.dilated[s], *be, Fmt::R8Unorm, G.quadW, G.quadH, "srsr.dilated");
        }
    }
    ok = ok && mk(R.guideBase, *be, Fmt::RGBA16Float, G.quadW, G.quadH, "srsr.guide.base");
    ok = ok && mk(R.guideScratch, *be, Fmt::RGBA16Float, G.quadW, G.quadH, "srsr.guide.alt");
    ok = ok && mk(R.rejection, *be, Fmt::R8Unorm, G.quadW, G.quadH, "srsr.rejection");
    ok = ok && mk(R.dilatePartial, *be, Fmt::R32Float, G.quadW, G.quadH, "srsr.dilatePartial");
    ok = ok && mk(R.unblocker, *be, Fmt::R8Unorm, G.unbW, G.unbH, "srsr.unblocker");
    ok = ok && mk(R.accW, *be, Fmt::R8Unorm, G.quadW, G.quadH, "srsr.accW");
    ok = ok && mk(R.dummyR8, *be, Fmt::R8Unorm, 1, 1, "srsr.dummy8");
    ok = ok && mk(R.dummyFlow, *be, Fmt::RGBA32Float, 1, 1, "srsr.dummyFlow");
    for (size_t c = 0; c < G.chain.size() && ok; ++c) {
        R.refChain.emplace_back(); R.movChain.emplace_back();
        ok = ok && mk(R.refChain.back(), *be, Fmt::R32Float, G.chain[c].first, G.chain[c].second, "srsr.refChain");
        ok = ok && mk(R.movChain.back(), *be, Fmt::R32Float, G.chain[c].first, G.chain[c].second, "srsr.movChain");
    }
    for (int l = 0; l < kLevels && ok; ++l) {
        const LevelGeo& lv = G.levels[l];
        R.gx.emplace_back(); R.gy.emplace_back(); R.hess.emplace_back();
        R.flowIn.emplace_back(); R.flowA.emplace_back(); R.flowB.emplace_back();
        ok = ok && mk(R.gx.back(), *be, Fmt::R32Float, lv.w, lv.h, "srsr.gx");
        ok = ok && mk(R.gy.back(), *be, Fmt::R32Float, lv.w, lv.h, "srsr.gy");
        ok = ok && mk(R.hess.back(), *be, Fmt::RGBA32Float, lv.tx(), lv.ty(), "srsr.hess");
        ok = ok && mk(R.flowIn.back(), *be, Fmt::RGBA32Float, lv.tx(), lv.ty(), "srsr.flowIn");
        ok = ok && mk(R.flowA.back(), *be, Fmt::RGBA32Float, lv.tx(), lv.ty(), "srsr.flowA");
        ok = ok && mk(R.flowB.back(), *be, Fmt::RGBA32Float, lv.tx(), lv.ty(), "srsr.flowB");
    }
    ok = ok && mk(R.flowFinal, *be, Fmt::RGBA32Float, G.flowTX, G.flowTY, "srsr.flowFinal");
    ok = ok && mk(R.flowMedian, *be, Fmt::RGBA32Float, G.flowTX, G.flowTY, "srsr.flowMedian");
    const size_t raw_bytes = (size_t)ref.w * ref.h * sizeof(float);
    for (int i = 0; i < 2 && ok; ++i) {
        R.staging[i] = be->make_buf(raw_bytes, "srsr.staging");
        ok = R.staging[i] != nullptr;
    }
    if (!ok) return fail("GPU allocation failed");
    prof_mark_memory("srsr:allocated");

    Run run{*be, G, T, work, R};
    const FrameNoise base_noise = frame_noise(work, 1.f);
    std::vector<SRSRFrameGeometry> fgeo;
    std::vector<SRSRFrameNormalization> fnorm;
    auto add_slot = [&](bool is_base) {
        SRSRFrameGeometry g{};
        g.flowScaleX = G.flowScaleX; g.flowScaleY = G.flowScaleY;
        g.flowOffsetX = G.flowOffsetX; g.flowOffsetY = G.flowOffsetY;
        g.isBase = is_base ? 1 : 0;
        fgeo.push_back(g);
        fnorm.push_back(quad_norm());
    };

    // ---- base frame: PREPARE, reference pyramid + gradients, base guide, unblocker
    SRSRNoiseLUT base_lut;
    t0 = prof_now_ms();
    build_noise_lut(base_noise, base_noise, T.greenDifferenceImpact, base_lut);
    prof_add_cpu("srsr:noise-lut", prof_now_ms() - t0);
    std::memcpy(R.staging[0]->contents(), ref.data.data(), raw_bytes);
    ref = Image();
    if (!be->begin()) return fail("GPU command buffer failed");
    run.prepare(0, R.staging[0].get(), R.refChain[0].get());
    run.downsample_chain(R.refChain);
    run.reference_gradients();
    run.guide(0, R.guideBase.get(), base_lut);
    if (work.srsr_unblocker) run.unblocker_map();
    uint64_t ticket_slot[2] = {be->commit("srsr:base"), 0};
    add_slot(true);
    report("SRSR: reference analysed", 0.08f);

    // ---- comparison frames: decode k+1 on a worker while the GPU analyses k
    struct Loaded { Image im; float gain = 1.f; };
    auto load = [&](int k) {
#if defined(__APPLE__)
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
        Loaded L;
        Config c = work;
        const double td = prof_now_ms();
        L.im = loader(k, c, false, G.rawH, G.rawW);
        if (L.im.h == G.rawH && L.im.w == G.rawW && L.im.c == 1 && cfg.exposure_match_enabled &&
            exp_ref > 0.f && L.im.exposure > 0.f) {
            const float g = std::min(1.5f, std::max(0.67f, exp_ref / L.im.exposure));
            if (std::fabs(g - 1.f) > 0.002f) {
                apply_raw_gain(L.im, g, sat);
                L.gain = g;
            }
        }
        prof_add_cpu("srsr:decode", prof_now_ms() - td);
        return L;
    };
    std::future<Loaded> next;
    if (n > 1) next = std::async(std::launch::async, load, 1);
    int slots = 1;
    int stage_i = 1;
    for (int k = 1; k < n; ++k) {
        Loaded L = next.get();
        if (k + 1 < n) next = std::async(std::launch::async, load, k + 1);
        if (L.im.h != G.rawH || L.im.w != G.rawW || L.im.c != 1) {
            report("SRSR: frame " + std::to_string(k) + " skipped (decode / size mismatch)", 0.08f + 0.62f * k / n);
            continue;
        }
        const int slot = slots;
        SRSRNoiseLUT lut;
        t0 = prof_now_ms();
        build_noise_lut(base_noise, frame_noise(work, L.gain), T.greenDifferenceImpact, lut);
        prof_add_cpu("srsr:noise-lut", prof_now_ms() - t0);
        const int st = stage_i & 1;
        if (ticket_slot[st] && !be->wait(ticket_slot[st])) return fail("GPU analysis failed");
        t0 = prof_now_ms();
        std::memcpy(R.staging[st]->contents(), L.im.data.data(), raw_bytes);
        L.im = Image();
        prof_add_cpu("srsr:upload", prof_now_ms() - t0);
        if (!be->begin()) return fail("GPU command buffer failed");
        run.prepare(slot, R.staging[st].get(), R.movChain[0].get());
        run.downsample_chain(R.movChain);
        run.align(slot);
        run.guide(slot, R.guideScratch.get(), lut);
        run.rejection_and_dilate(slot, lut);
        ticket_slot[st] = be->commit("srsr:frame");
        ++stage_i;
        add_slot(false);
        ++slots;
        prof_mark_memory("srsr:frame");
        report("SRSR: aligned frame " + std::to_string(k + 1) + " / " + std::to_string(n), 0.08f + 0.62f * (float)(k + 1) / n);
    }
    if (!be->wait_all()) return fail("GPU analysis failed");
    R.free_analysis();
    prof_mark_memory("srsr:analysis-freed");

    // ---- COPY_MASK
    const int nslots = slots;
    {
        SRSRAccumulateUniforms u{G.quadW, G.quadH, nslots - 1, T.accumulatedWeightScale};
        if (!be->begin()) return fail("GPU command buffer failed");
        for (int f = 0; f < (int)kSRSRMaxFrames; ++f) {
            const int s = 1 + f;
            be->tex(f, s < nslots ? R.dilated[s].get() : R.dummyR8.get());
        }
        be->tex((int)kSRSRMaxFrames, R.accW.get());
        be->bytes(0, &u, sizeof u);
        be->dispatch(K::accumulate_weights, G.quadW, G.quadH);
        // Same bindings: the dilated masks stay at 0..15 and the uniforms at 0.
        if (work.robustness_save_mask) {
            R.maskOut = be->make_buf((size_t)G.quadW * G.quadH, "srsr.mask");
            if (R.maskOut) {
                be->buf(1, R.maskOut.get());
                be->dispatch(K::mask_export, G.quadW, G.quadH);
            }
        }
        be->commit("srsr:copy-mask");
    }

    // ---- output
    const int Hs = G.outH, Ws = G.outW;
    const int band_rows = std::min(band, Hs);
    if (!mk(R.bandA, *be, Fmt::RGBA16Float, Ws, band_rows + 2 * halo, "srsr.bandA") ||
        !mk(R.bandB, *be, Fmt::RGBA16Float, Ws, band_rows + 2 * halo, "srsr.bandB"))
        return fail("GPU allocation failed (bands)");
    for (int i = 0; i < 2; ++i) {
        R.out[i] = be->make_buf((size_t)band_rows * Ws * 3 * sizeof(uint16_t), "srsr.out");
        if (!R.out[i]) return fail("GPU allocation failed (output)");
    }
    prof_mark_memory("srsr:bands");

    DngStreamWriter writer;
    const std::string& model = work.camera_model.empty() ? std::string("HandheldSR-x2") : work.camera_model;
    const std::string& make = work.camera_make.empty() ? std::string("HandheldSR") : work.camera_make;
    const CaptureExif dng_exif = DngStreamWriter::exif_from_config(work);
    if (!writer.open(dng_path, Ws, Hs, model, work.orientation,
                     work.has_color_matrix ? work.color_matrix : nullptr,
                     work.white_balance, work.bake_srgb, make,
                     work.has_cam_to_srgb ? work.cam_to_srgb : nullptr,
                     work.raw_prewhitened && !dng_unwhiten_active(work, 3),
                     work.dng_codec, &dng_exif, work.num_threads))
        return fail("cannot open output DNG");
    float sg[3];
    dng_unwhiten_gains(work, 3, sg);
    const float pscale = std::min(1.f, (float)maxPreviewDim / (float)std::max(Hs, Ws));
    const int ph = std::max(1, (int)(Hs * pscale)), pw = std::max(1, (int)(Ws * pscale));
    Image preview(ph, pw, 3);

    float yuv[4];
    yuv_noise_model(base_noise, yuv);
    SRSRMergeUniforms mu{};
    mu.outWidth = Ws; mu.outHeight = Hs; mu.bayerWidth = G.bayerW; mu.bayerHeight = G.bayerH;
    mu.quadWidth = G.quadW; mu.quadHeight = G.quadH;
    mu.frameCount = nslots; mu.baseIndex = 0; mu.applyRejectionWeight = 1;
    mu.coverageThreshold = T.coverageThreshold; mu.covarianceMultiplier = T.covarianceMultiplier;
    mu.accumulatedWeightScale = T.accumulatedWeightScale; mu.kernelEpsilon = T.kernelEpsilon;
    mu.normalization = T.normalization; mu.footprintLimit = T.footprintLimit;
    mu.borderMinX = G.borderMinX; mu.borderMinY = G.borderMinY;
    mu.borderMaxX = G.borderMaxX; mu.borderMaxY = G.borderMaxY;
    mu.dehomogenizeEpsilon = T.dehomogenizeEpsilon;

    auto encode_band = [&](int y0, int y1, Buf* outb) -> uint64_t {
        const int my0 = std::max(0, y0 - halo), my1 = std::min(Hs, y1 + halo);
        if (!be->begin()) return 0;
        mu.bandY0 = my0; mu.bandRows = my1 - my0;
        be->tex(0, R.bandA.get());
        be->tex(1, R.accW.get());
        for (int f = 0; f < (int)kSRSRMaxFrames; ++f) {
            const int s = f < nslots ? f : 0;
            be->tex(8 + f, R.quad[s].get());
            be->tex(24 + f, R.cov[s].get());
            be->tex(40 + f, s > 0 ? R.flow[s].get() : R.dummyFlow.get());
            be->tex(56 + f, s > 0 ? R.dilated[s].get() : R.dummyR8.get());
        }
        be->bytes(0, &mu, sizeof mu);
        be->bytes(1, fnorm.data(), fnorm.size() * sizeof(SRSRFrameNormalization));
        be->bytes(2, fgeo.data(), fgeo.size() * sizeof(SRSRFrameGeometry));
        be->dispatch(K::merge, Ws, my1 - my0);

        SRSRDenoiseUniforms du{};
        du.width = Ws; du.height = Hs;
        du.texY0 = my0; du.texRows = my1 - my0;
        du.weightTextureScale = T.weightTextureScale; du.weightTextureBias = T.weightTextureBias;
        du.denoisingFactor = T.denoisingFactor; du.outlierSigmaMultiplier = T.outlierSigmaMultiplier;
        du.rejectLumaRevertFactor = T.rejectLumaRevertFactor; du.globalLumaRevertFactor = T.globalLumaRevertFactor;
        du.vignetteStrengthBoost = T.vignetteStrengthBoost; du.vignetteSpatialScaleFactor = T.vignetteSpatialScaleFactor;
        du.yuvNoiseModelX = yuv[0]; du.yuvNoiseModelY = yuv[1]; du.yuvNoiseModelZ = yuv[2]; du.yuvNoiseModelW = yuv[3];
        du.outGainR = sg[0]; du.outGainG = sg[1]; du.outGainB = sg[2];
        if (!work.srsr_denoise) {
            du.passthrough = 1; du.isLastLevel = 1;
            du.outY0 = y0; du.outRows = y1 - y0;
            be->tex(0, R.bandA.get()); be->tex(1, R.accW.get()); be->tex(2, R.bandB.get());
            be->buf(1, outb);
            be->bytes(0, &du, sizeof du);
            be->dispatch(K::denoise, Ws, y1 - y0);
        } else {
            Tex* src = R.bandA.get();
            Tex* dst = R.bandB.get();
            int remaining = 7 * dil_base;           // halo still needed below this level
            for (int level = 0; level < 3; ++level) {
                const int d = (1 << level) * dil_base;
                remaining -= d;
                const int oy0 = std::max(0, y0 - remaining), oy1 = std::min(Hs, y1 + remaining);
                du.level = level; du.dilation = d;
                du.isLastLevel = level == 2 ? 1 : 0;
                du.outY0 = oy0; du.outRows = oy1 - oy0;
                be->tex(0, src); be->tex(1, R.accW.get()); be->tex(2, dst);
                be->buf(1, outb);
                be->bytes(0, &du, sizeof du);
                be->dispatch(K::denoise, Ws, oy1 - oy0);
                std::swap(src, dst);
            }
        }
        return be->commit("srsr:band");
    };

    auto emit_band = [&](int y0, int y1, Buf* outb) {
        const uint16_t* rows = (const uint16_t*)outb->contents();
        const double tw = prof_now_ms();
        writer.write_rows(rows, y1 - y0);
        prof_add_cpu("srsr:dng-write", prof_now_ms() - tw);
        // Preview: nearest samples of the linear, white-balanced values.
        const int py0 = (int)std::ceil(y0 * pscale - 0.5f), py1 = (int)std::ceil(y1 * pscale - 0.5f);
        for (int py = std::max(0, py0); py < std::min(ph, py1); ++py) {
            const int y = std::min(Hs - 1, (int)((py + 0.5f) / pscale));
            if (y < y0 || y >= y1) continue;
            for (int px = 0; px < pw; ++px) {
                const int x = std::min(Ws - 1, (int)((px + 0.5f) / pscale));
                const uint16_t* p = rows + ((size_t)(y - y0) * Ws + x) * 3;
                srsr_preview_store_sample(preview, py, px, p[0] / 65535.f / sg[0], p[1] / 65535.f / sg[1],
                                     p[2] / 65535.f / sg[2], work, 3);
            }
        }
    };

    const int nb = (Hs + band_rows - 1) / band_rows;
    uint64_t prev_ticket = 0;
    int prev_y0 = 0, prev_y1 = 0;
    for (int b = 0; b < nb; ++b) {
        const int y0 = b * band_rows, y1 = std::min(Hs, y0 + band_rows);
        const uint64_t t = encode_band(y0, y1, R.out[b & 1].get());
        if (!t) return fail("GPU command buffer failed (merge)");
        if (b > 0) {
            if (!be->wait(prev_ticket)) return fail("GPU merge failed");
            emit_band(prev_y0, prev_y1, R.out[(b - 1) & 1].get());
        }
        prev_ticket = t; prev_y0 = y0; prev_y1 = y1;
        report("SRSR: merging", 0.72f + 0.26f * (float)y1 / Hs);
    }
    if (!be->wait(prev_ticket)) return fail("GPU merge failed");
    emit_band(prev_y0, prev_y1, R.out[(nb - 1) & 1].get());
    const double tc = prof_now_ms();
    writer.close();
    prof_add_cpu("srsr:dng-close", prof_now_ms() - tc);
    // The copy-mask batch was committed before the bands, so it is complete by
    // now on an in-order queue; waiting makes that explicit.
    if (R.maskOut && be->wait_all()) {
        // <stem>_robustness.pgm, the file the app picks up for "Save Robustness
        // Mask": mean keep over the comparison frames, nearest-upsampled from the
        // quad grid to the Bayer grid (raw resolution), as the standard pipeline.
        const double tm = prof_now_ms();
        Image m(G.quadH, G.quadW, 1);
        const uint8_t* src = (const uint8_t*)R.maskOut->contents();
        for (size_t i = 0; i < m.data.size(); ++i) m.data[i] = (f32)src[i] * (1.f / 255.f);
        R.maskOut.reset();
        if (write_robustness_mask_pgm(m, 1, dng_path, "", G.bayerH, G.bayerW))
            report("Wrote robustness mask", 0.99f);
        prof_add_cpu("srsr:mask-write", prof_now_ms() - tm);
    }
    prof_mark_memory("srsr:end");

    if (prof_enabled()) {
        const double wall = prof_now_ms() - t_burst;
        char hdr[256];
        std::snprintf(hdr, sizeof(hdr),
                      "\nSRSR (%s) burst wall %.1f ms over %d frames (%.1f ms/frame), %dx%d -> %dx%d, band %d rows, planned peak %.0f MB\n",
                      be->name(), wall, nslots, wall / std::max(1, nslots), G.bayerW, G.bayerH, Ws, Hs, band_rows,
                      planned_mb);
        std::string prof = prof_report() + hdr;
        std::printf("%s", prof.c_str());
        std::fflush(stdout);
        prof_save_report(prof);
        char one[192];
        std::snprintf(one, sizeof(one), "SRSR %.1fs total, %.0f ms/frame - peak %.0f MB (planned %.0f)",
                      wall / 1000.0, wall / std::max(1, nslots),
                      (double)prof_peak_footprint_bytes() / (1024.0 * 1024.0), planned_mb);
        report(one, 0.995f);
    }
    report("Done", 1.f);
    return preview;
}

} // namespace hhsr
