"""Wire ConfNet into the app (idempotent; run once, re-running is a no-op). Off by default everywhere.

NOT WIRED IN TODAY: shipped in 9910c0c, removed from the app at the owner's request (v21 stays the app's learned
mask). This script re-applies the integration and copies cpp/app/conf_nn_shared.h + conf_nn_weights.h into core/.

    python integrate_app.py

  core/types.h              Config::conf_nn_enabled, Config::conf_nn_combine
  core/HHSRKernels.metal    kernels cnn_inputs / cnn_enc / cnn_f1 / cnn_head (per-pixel code: core/conf_nn_shared.h)
  core/metal_gpu.mm         cnn_refine_raw_res(): banded like the robust_nn v15 path, runs on the robustness mask's
                            guide grid after the analytic mask, before robust_nn v14/v15 (which it then replaces);
                            reference grey pinned when either NN is on
  Sources/CameraModel.swift TuningParams.conf_nn_enabled / conf_nn_combine (+ coding keys, decoding, bridge dict)
  Sources/Bridge/SRBridge.mm  tuning -> Config
  Sources/CameraView.swift  Settings toggle + combine picker in the live Robustness section
The device path runs the per-frame model form only (CNN_FUSION 0, CNN_UP 0); with a fused or raw-resolution
weights header it declines and the analytic mask is used, exactly as when the toggle is off.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARK = "conf_nn"


def patch(rel, pairs, marker):
    p = os.path.join(ROOT, rel)
    s = open(p, encoding="utf-8").read()
    if marker in s:
        print("already patched", rel)
        return
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, f"{rel}: anchor found {n} times: {old[:120]!r}"
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(s)
    print("patched", rel)


TYPES = ('''    bool  robust_nn_v15_enabled = false;
''', '''    bool  robust_nn_v15_enabled = false;
    // ConfNet (tools/mfsr_conf, opt-in): per-frame confidence network trained against the app's own merge on
    // synthetic bursts with exact ground truth (reconstruction-optimal labels, a misalignment priced 4x the same
    // amount of noise). Runs on the robustness mask's guide grid after the analytic mask (banded kernels cnn_*,
    // core/conf_nn_shared.h + conf_nn_weights.h, ~30 MB of band buffers on a 12 MP burst) and takes precedence over
    // robust_nn v14 / v15 when both are on. Only the per-frame form (no cross-frame fusion, guide-resolution output)
    // runs on device; with any other weights header it declines and the analytic mask is kept. Off by default.
    bool  conf_nn_enabled = false;
    // How the network's weight combines with the analytic mask: 0 = network alone, 1 = min(network, analytic),
    // 2 = network x analytic.
    int   conf_nn_combine = 0;
''')

METAL_INC = ('''#include "robust_nn_shared.h"
''', '''#include "robust_nn_shared.h"
#include "conf_nn_shared.h"
''')

METAL_KERNELS_ANCHOR = '''    r_out[y * W + x] = R;
}
'''
METAL_KERNELS = METAL_KERNELS_ANCHOR + r'''

// ===========================================================================
// ConfNet (Config::conf_nn_enabled, tools/mfsr_conf): per-frame robustness weight from a compact CNN trained
// against the app's own merge. Banded exactly like the rnn15_* path above (same warp and 14-feature kernels, same
// row-band buffers), then: 18 inputs -> 4 dilated 3x3 convs (dilations CNN_DIL0..3) -> 1x1 -> 3x3 -> head -> R.
// Per-pixel code in conf_nn_shared.h (parity-tested on CPU against PyTorch: tools/mfsr_conf/parity_cnn.py).
// Mirrors CnnBandCPU in metal_gpu.mm (64 bytes).
struct CnnBand {
    int lo, hi;                 // absolute rows this dispatch computes
    int jb, jn, fb, fn, xb, xn; // band buffers (warped grey, features, inputs): first row, row count
    int ib, in_, ob, on;        // layer input / output planes: first row, row count
    int layer, dil;             // encoder layer 0..3 (cnn_enc)
    int combine, _pad1;         // cnn_head: 0 network, 1 min(network, analytic), 2 network x analytic
};

kernel void cnn_inputs(device const float* ref_grey [[buffer(0)]],
                       device const float* jband [[buffer(1)]],
                       device const float* fband [[buffer(2)]],
                       device float* xband [[buffer(3)]],
                       constant RnnAppParams& p [[buffer(4)]],
                       constant CnnBand& b [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const int W = p.width, x = int(gid.x), y = b.lo + int(gid.y);
    if (x >= W || y >= b.hi) return;
    RNNParams P = {W, p.height, p.flow_step, p.noise_a, p.noise_b, p.clip_level, 0, 0};
    RNNImg rf = {ref_grey, W};
    Rnn15Rows tj = {jband, W, b.jb};
    Rnn15Planes tf = {fband, W, b.fb, b.fn};
    float xin[CNN_NG];
    cnn_inputs_at(rf, tj, tf, P, x, y, xin);
    for (int c = 0; c < CNN_NG; ++c) xband[(c * b.xn + (y - b.xb)) * W + x] = xin[c];
}

kernel void cnn_enc(device const float* in [[buffer(0)]],
                    device float* out [[buffer(1)]],
                    constant RnnAppParams& p [[buffer(2)]],
                    constant CnnBand& b [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    const int W = p.width, x = int(gid.x), y = b.lo + int(gid.y);
    if (x >= W || y >= b.hi) return;
    Rnn15Planes ti = {in, W, b.ib, b.in_};
    float o[CNN_CH];
    cnn_encoder_layer_at(ti, b.layer, W, p.height, x, y, o);
    for (int c = 0; c < CNN_CH; ++c) out[(c * b.on + (y - b.ob)) * W + x] = o[c];
}

kernel void cnn_f1(device const float* in [[buffer(0)]],
                   device float* out [[buffer(1)]],
                   constant RnnAppParams& p [[buffer(2)]],
                   constant CnnBand& b [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    const int W = p.width, x = int(gid.x), y = b.lo + int(gid.y);
    if (x >= W || y >= b.hi) return;
    float fk[CNN_CH], o[CNN_CH];
    for (int c = 0; c < CNN_CH; ++c) fk[c] = in[(c * b.in_ + (y - b.ib)) * W + x];
    cnn_fuse1(fk, fk, fk, o);       // per-frame form: the host only runs this kernel when CNN_FUSION == 0
    for (int c = 0; c < CNN_CH; ++c) out[(c * b.on + (y - b.ob)) * W + x] = o[c];
}

kernel void cnn_head(device const float* g1 [[buffer(0)]],
                     device const float* r0 [[buffer(1)]],
                     device float* r_out [[buffer(2)]],
                     constant RnnAppParams& p [[buffer(3)]],
                     constant CnnBand& b [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const int W = p.width, x = int(gid.x), y = b.lo + int(gid.y);
    if (x >= W || y >= b.hi) return;
    Rnn15Planes tg = {g1, W, b.ib, b.in_};
    float hid[CNN_CH], lg[2];
    cnn_head_at(tg, W, p.height, x, y, hid, lg);
    const float r0v = r0[y * W + x];
    float R = cnn_weight(lg[1], r0v);
    if (b.combine == 1) R = min(R, r0v);
    else if (b.combine == 2) R = R * r0v;
    r_out[y * W + x] = R;
}
'''

MM_INC = ('''#include "robust_nn_shared.h"   // RNN_NF / RNN15_* sizes for the v15 band buffers
''', '''#include "robust_nn_shared.h"   // RNN_NF / RNN15_* sizes for the v15 band buffers
#include "conf_nn_shared.h"     // CNN_* sizes for the ConfNet band buffers (Config::conf_nn_enabled)
''')

MM_GLOBALS = ('''static id<MTLBuffer> g_r15_j = nil, g_r15_f = nil, g_r15_x = nil, g_r15_a = nil, g_r15_b = nil;
''', '''static id<MTLBuffer> g_r15_j = nil, g_r15_f = nil, g_r15_x = nil, g_r15_a = nil, g_r15_b = nil;
// ConfNet band buffers (Config::conf_nn_enabled): same lifetime as the v15 ones.
static id<MTLBuffer> g_cnn_j = nil, g_cnn_f = nil, g_cnn_x = nil, g_cnn_a = nil, g_cnn_b = nil;
''')

MM_RESET = ('''    g_r15_j = g_r15_f = g_r15_x = g_r15_a = g_r15_b = nil;
}
''', '''    g_r15_j = g_r15_f = g_r15_x = g_r15_a = g_r15_b = nil;
    g_cnn_j = g_cnn_f = g_cnn_x = g_cnn_a = g_cnn_b = nil;
}
''')

MM_PIN = ('''static void rnn_pin_ref_grey(id<MTLBuffer> b_guide, int gh, int gw, int nch, const Config& cfg,
                             id<MTLCommandBuffer> cmd) {
    if (!rnn_domain_ok(cfg, nch)) return;''', '''static bool cnn_domain_ok(const Config& cfg, int nch) {
    return cfg.conf_nn_enabled && nch == 3 && !cfg.robustness_perceptual_sqrt;
}

static void rnn_pin_ref_grey(id<MTLBuffer> b_guide, int gh, int gw, int nch, const Config& cfg,
                             id<MTLCommandBuffer> cmd) {
    if (!rnn_domain_ok(cfg, nch) && !cnn_domain_ok(cfg, nch)) return;   // either network reads the reference grey''')

MM_FUNC_ANCHOR = '''static RefStats init_robustness_metal_impl(const Image& ref_raw, const Config& cfg) {'''
MM_FUNC = r'''// ---- ConfNet (Config::conf_nn_enabled, tools/mfsr_conf), banded (kernels rnn15_warp / rnn15_feat / cnn_*) --------
// Mirrors CnnBand in HHSRKernels.metal.
struct CnnBandCPU {
    int32_t lo, hi, jb, jn, fb, fn, xb, xn, ib, in_, ob, on, layer, dil, combine, _pad1;
};
static_assert(sizeof(CnnBandCPU) == 64, "CnnBandCPU must mirror CnnBand");
static_assert(sizeof(CnnBandCPU) == sizeof(Rnn15BandCPU), "the warp / feature kernels read the band as Rnn15Band");

// Output rows per band, and the rows each stage needs beyond the band per side. The head's 3x3 reads f1 at +-1; f1 is
// 1x1 on encoder layer 4 (+-1); layer 4 (dilation d3) reads layer 3 at +-(1 + d3); and so on down to the inputs;
// the inputs read the features at the same pixel; the features read the warped grey up to +-4.
static const int kCnnBand = 32;

// Same contract as rnn15_refine_raw_res (same inputs, output written back quad by quad). Returns false (R untouched)
// whenever it cannot run: toggle off, wrong domain, a weights header that is not the per-frame form, missing kernel.
static bool cnn_refine_raw_res(Image& R, id<MTLBuffer> b_comp_guide, int gh, int gw, int nch,
                               const FlowField& flow, int tile_size, const Config& cfg,
                               id<MTLBuffer> b_flow, id<MTLBuffer> b_jac, bool use_affine) {
    if (!cnn_domain_ok(cfg, nch) || cfg.dense_flow_enabled) return false;
    if (CNN_FUSION != 0 || CNN_UP != 0) return false;       // device runs the per-frame, guide-resolution form only
    if (!g_rnn_ref_grey || g_rnn_h != gh || g_rnn_w != gw || !b_comp_guide) return false;
    if (R.c != 1 || R.h < 1 || R.w < 1 || R.data.empty()) return false;
    if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty() || tile_size <= 0 || !b_flow || !b_jac) return false;
    auto& c = ctx();
    id<MTLComputePipelineState> pw = c.pipe("rnn15_warp"), pf = c.pipe("rnn15_feat"), pi = c.pipe("cnn_inputs"),
                                pe = c.pipe("cnn_enc"), p1 = c.pipe("cnn_f1"), ph = c.pipe("cnn_head");
    if (!pw || !pf || !pi || !pe || !p1 || !ph) return false;
    const int d[4] = {CNN_DIL0, CNN_DIL1, CNN_DIL2, CNN_DIL3};
    // halo of each stage's OUTPUT rows (per side): head 0, f1 1, enc4 1, enc3 1 + d3, enc2 + d2, enc1 + d1,
    // inputs + d0, features = inputs, warped grey = features + 4
    const int h_f1 = 1, h_e4 = 1, h_e3 = h_e4 + d[3], h_e2 = h_e3 + d[2], h_e1 = h_e2 + d[1], h_x = h_e1 + d[0];
    const int h_f = h_x, h_j = h_f + 4;
    const int h_a = h_e1 > h_e3 ? h_e1 : h_e3;               // plane A holds enc1, enc3 and f1
    const int h_b = h_e2 > h_e4 ? h_e2 : h_e4;               // plane B holds enc2 and enc4
    const size_t n = (size_t)gh * (size_t)gw, bytes = n * sizeof(float);
    if (!g_rnn_r0 || [g_rnn_r0 length] < bytes) g_rnn_r0 = buf(nullptr, bytes);
    if (!g_rnn_out || [g_rnn_out length] < bytes) g_rnn_out = buf(nullptr, bytes);
    const int B = kCnnBand;
    const size_t row = (size_t)gw * sizeof(float);
    const size_t need_j = (size_t)(B + 2 * h_j) * row, need_f = (size_t)RNN_NF * (B + 2 * h_f) * row,
                 need_x = (size_t)CNN_NG * (B + 2 * h_x) * row, need_a = (size_t)CNN_CH * (B + 2 * h_a) * row,
                 need_b = (size_t)CNN_CH * (B + 2 * h_b) * row;
    if (!g_cnn_j || [g_cnn_j length] < need_j) g_cnn_j = buf(nullptr, need_j);
    if (!g_cnn_f || [g_cnn_f length] < need_f) g_cnn_f = buf(nullptr, need_f);
    if (!g_cnn_x || [g_cnn_x length] < need_x) g_cnn_x = buf(nullptr, need_x);
    if (!g_cnn_a || [g_cnn_a length] < need_a) g_cnn_a = buf(nullptr, need_a);
    if (!g_cnn_b || [g_cnn_b length] < need_b) g_cnn_b = buf(nullptr, need_b);
    if (!g_rnn_r0 || !g_rnn_out || !g_cnn_j || !g_cnn_f || !g_cnn_x || !g_cnn_a || !g_cnn_b) return false;

    float* r0 = (float*)[g_rnn_r0 contents];
    for (int y = 0; y < gh; ++y) {
        const int ry = std::min(2 * y, R.h - 1);
        for (int x = 0; x < gw; ++x)
            r0[(size_t)y * gw + x] = R.data[(size_t)ry * R.w + std::min(2 * x, R.w - 1)];
    }
    RnnAppParamsCPU p{};
    p.width = gw;
    p.height = gh;
    p.flow_step = std::max(1, tile_size / 2);
    p.noise_a = (cfg.alpha_dng[0] + 2.f * cfg.alpha_dng[1] + cfg.alpha_dng[2]) / 16.f;
    p.noise_b = (cfg.beta_dng[0] + 2.f * cfg.beta_dng[1] + cfg.beta_dng[2]) / 16.f;
    p.clip_level = 0.9f;
    p.flow_ny = (uint32_t)flow.ny;
    p.flow_nx = (uint32_t)flow.nx;
    p.tile_size = (float)tile_size;
    p.affine = use_affine ? 1u : 0u;
    p.homog = flow.has_global_h ? 1u : 0u;
    for (int i = 0; i < 9; ++i) p.hmat[i] = flow.global_h[i];
    p.sharp_lo = 0.f;
    p.sharp_hi = 0.f;

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = cmd ? [cmd computeCommandEncoder] : nil;
    if (!enc) return false;
    auto rows = [&](int a, int b) { return std::max(0, std::min(gh, b) - std::max(0, a)); };
    for (int y0 = 0; y0 < gh; y0 += B) {
        const int y1 = std::min(gh, y0 + B);
        CnnBandCPU bd{};
        bd.combine = std::min(std::max(cfg.conf_nn_combine, 0), 2);
        bd.jb = std::max(0, y0 - h_j); bd.jn = rows(y0 - h_j, y1 + h_j);
        bd.fb = std::max(0, y0 - h_f); bd.fn = rows(y0 - h_f, y1 + h_f);
        bd.xb = std::max(0, y0 - h_x); bd.xn = rows(y0 - h_x, y1 + h_x);
        const int ab = std::max(0, y0 - h_a), an = rows(y0 - h_a, y1 + h_a);
        const int bb = std::max(0, y0 - h_b), bn = rows(y0 - h_b, y1 + h_b);
        auto stage = [&](id<MTLComputePipelineState> ps, int band_index, int lo, int hi) {
            bd.lo = std::max(0, lo);
            bd.hi = std::min(gh, hi);
            [enc setBytes:&bd length:sizeof(bd) atIndex:band_index];
            dispatch2(enc, ps, (NSUInteger)gw, (NSUInteger)(bd.hi - bd.lo));
        };
        // 1) warped comparison grey (the merge's correspondence)
        [enc setBuffer:b_comp_guide offset:0 atIndex:0];
        [enc setBuffer:g_cnn_j offset:0 atIndex:1];
        [enc setBuffer:b_flow offset:0 atIndex:2];
        [enc setBuffer:b_jac offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        stage(pw, 5, y0 - h_j, y1 + h_j);
        // 2) the 14 features
        [enc setBuffer:g_rnn_ref_grey offset:0 atIndex:0];
        [enc setBuffer:g_cnn_j offset:0 atIndex:1];
        [enc setBuffer:g_rnn_r0 offset:0 atIndex:2];
        [enc setBuffer:g_cnn_f offset:0 atIndex:3];
        [enc setBuffer:b_flow offset:0 atIndex:4];
        [enc setBuffer:b_jac offset:0 atIndex:5];
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        stage(pf, 7, y0 - h_f, y1 + h_f);
        // 3) the 18 inputs
        [enc setBuffer:g_rnn_ref_grey offset:0 atIndex:0];
        [enc setBuffer:g_cnn_j offset:0 atIndex:1];
        [enc setBuffer:g_cnn_f offset:0 atIndex:2];
        [enc setBuffer:g_cnn_x offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        stage(pi, 5, y0 - h_x, y1 + h_x);
        // 4) encoder: X -> A (enc1) -> B (enc2) -> A (enc3) -> B (enc4)
        const int halo_out[4] = {h_e1, h_e2, h_e3, h_e4};
        for (int l = 0; l < 4; ++l) {
            const bool from_x = (l == 0), to_a = (l % 2 == 0);
            bd.layer = l;
            bd.dil = d[l];
            bd.ib = from_x ? bd.xb : (to_a ? bb : ab);
            bd.in_ = from_x ? bd.xn : (to_a ? bn : an);
            bd.ob = to_a ? ab : bb;
            bd.on = to_a ? an : bn;
            [enc setBuffer:(from_x ? g_cnn_x : (to_a ? g_cnn_b : g_cnn_a)) offset:0 atIndex:0];
            [enc setBuffer:(to_a ? g_cnn_a : g_cnn_b) offset:0 atIndex:1];
            [enc setBytes:&p length:sizeof(p) atIndex:2];
            stage(pe, 3, y0 - halo_out[l], y1 + halo_out[l]);
        }
        // 5) f1 (1x1): B (enc4) -> A
        bd.ib = bb; bd.in_ = bn; bd.ob = ab; bd.on = an;
        [enc setBuffer:g_cnn_b offset:0 atIndex:0];
        [enc setBuffer:g_cnn_a offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        stage(p1, 3, y0 - h_f1, y1 + h_f1);
        // 6) f2 (3x3) + head + combination with the analytic mask -> output plane
        bd.ib = ab; bd.in_ = an;
        [enc setBuffer:g_cnn_a offset:0 atIndex:0];
        [enc setBuffer:g_rnn_r0 offset:0 atIndex:1];
        [enc setBuffer:g_rnn_out offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        stage(ph, 4, y0, y1);
    }
    [enc endEncoding];
    prof_tag_gpu(cmd, "robustness:conf_nn");
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return false;

    const float* out = (const float*)[g_rnn_out contents];
    for (int y = 0; y < R.h; ++y) {
        const float* orow = out + (size_t)std::min(y / 2, gh - 1) * gw;
        float* dst = R.data.data() + (size_t)y * R.w;
        for (int x = 0; x < R.w; ++x) dst[x] = orow[std::min(x / 2, gw - 1)];
    }
    return true;
}

''' + MM_FUNC_ANCHOR

MM_CALL = ('''    if (rob_nch == 1u &&
        !rnn15_refine_raw_res(r, b_guide, gh, gw, nch, flow, tile_size, cfg, b_geom_flow, b_geom_affine,
                              raw_geom_affine))
        rnn_refine_raw_res(r, b_guide, gh, gw, nch, flow, tile_size, cfg, b_geom_flow, b_geom_affine,
                           raw_geom_affine);''', '''    // ConfNet (Config::conf_nn_enabled) first; when it runs it replaces v14 / v15.
    if (rob_nch == 1u &&
        !cnn_refine_raw_res(r, b_guide, gh, gw, nch, flow, tile_size, cfg, b_geom_flow, b_geom_affine,
                            raw_geom_affine) &&
        !rnn15_refine_raw_res(r, b_guide, gh, gw, nch, flow, tile_size, cfg, b_geom_flow, b_geom_affine,
                              raw_geom_affine))
        rnn_refine_raw_res(r, b_guide, gh, gw, nch, flow, tile_size, cfg, b_geom_flow, b_geom_affine,
                           raw_geom_affine);''')

SW_DECL = ('''    var robust_nn_v15_enabled: Bool = false
''', '''    var robust_nn_v15_enabled: Bool = false
    /// ConfNet confidence mask (tools/mfsr_conf): per-frame robustness weight from a network trained against the
    /// app's own merge. Off by default. Combine: 0 network alone, 1 min(network, analytic), 2 network x analytic.
    var conf_nn_enabled: Bool = false
    var conf_nn_combine: Int = 0
''')
SW_KEYS = ('''        case robust_nn_v15_enabled
''', '''        case robust_nn_v15_enabled
        case conf_nn_enabled, conf_nn_combine
''')
SW_DECODE = ('''        robust_nn_v15_enabled = try c.decodeIfPresent(Bool.self, forKey: .robust_nn_v15_enabled) ?? robust_nn_v15_enabled
''', '''        robust_nn_v15_enabled = try c.decodeIfPresent(Bool.self, forKey: .robust_nn_v15_enabled) ?? robust_nn_v15_enabled
        conf_nn_enabled = try c.decodeIfPresent(Bool.self, forKey: .conf_nn_enabled) ?? conf_nn_enabled
        conf_nn_combine = try c.decodeIfPresent(Int.self, forKey: .conf_nn_combine) ?? conf_nn_combine
''')
SW_DICT = ('''            "robust_nn_v15_enabled": NSNumber(value: tuningParams.robust_nn_v15_enabled),
''', '''            "robust_nn_v15_enabled": NSNumber(value: tuningParams.robust_nn_v15_enabled),
            "conf_nn_enabled": NSNumber(value: tuningParams.conf_nn_enabled),
            "conf_nn_combine": NSNumber(value: tuningParams.conf_nn_combine),
''')
BRIDGE = ('''    if (tuning[@"robust_nn_v15_enabled"])
        cfg.robust_nn_v15_enabled = tuning[@"robust_nn_v15_enabled"].boolValue;
''', '''    if (tuning[@"robust_nn_v15_enabled"])
        cfg.robust_nn_v15_enabled = tuning[@"robust_nn_v15_enabled"].boolValue;
    if (tuning[@"conf_nn_enabled"])
        cfg.conf_nn_enabled = tuning[@"conf_nn_enabled"].boolValue;
    if (tuning[@"conf_nn_combine"])
        cfg.conf_nn_combine = std::max(0, std::min(2, tuning[@"conf_nn_combine"].intValue));
''')
VIEW = ('''                    Toggle("Robust NN v21 (GPU)", isOn: $cam.tuningParams.robust_nn_enabled)
''', '''                    Toggle("ConfNet confidence mask", isOn: $cam.tuningParams.conf_nn_enabled)
                    Text(confNetHelp)
                        .font(.footnote).foregroundColor(.secondary)
                    if cam.tuningParams.conf_nn_enabled {
                        Picker("Combine with analytic mask", selection: $cam.tuningParams.conf_nn_combine) {
                            Text("Network only").tag(0)
                            Text("Minimum").tag(1)
                            Text("Product").tag(2)
                        }
                    }
                    Toggle("Robust NN v21 (GPU)", isOn: $cam.tuningParams.robust_nn_enabled)
''')
VIEW_HELP = ('''    private var chooseReferenceHelp: String {''', '''    // Out of the ViewBuilder for the same type-checker reason as chooseReferenceHelp.
    private var confNetHelp: String {
        return "Per-frame robustness weight (0 = sample discarded, 1 = fully merged) from a compact network (tools/mfsr_conf) trained against this app's own merge on synthetic bursts with exact ground truth: it keeps correctly aligned samples, aliased detail and flat noisy areas, and drops samples whose misalignment would paint a ghost or a doubled edge in the merged image. Replaces Robust NN v21/v39 while on. Off by default."
    }

    private var chooseReferenceHelp: String {''')


def main():
    import shutil
    for h in ("conf_nn_shared.h", "conf_nn_weights.h"):
        shutil.copy(os.path.join(ROOT, "tools", "mfsr_conf", "cpp", "app", h), os.path.join(ROOT, "core", h))
    patch("core/types.h", [TYPES], "conf_nn_enabled")
    patch("core/HHSRKernels.metal", [METAL_INC, (METAL_KERNELS_ANCHOR, METAL_KERNELS)], "cnn_inputs")
    patch("core/metal_gpu.mm", [MM_INC, MM_GLOBALS, MM_RESET, MM_PIN, (MM_FUNC_ANCHOR, MM_FUNC), MM_CALL],
          "cnn_refine_raw_res")
    patch("Sources/CameraModel.swift", [SW_DECL, SW_KEYS, SW_DECODE, SW_DICT], "conf_nn_enabled")
    patch("Sources/Bridge/SRBridge.mm", [BRIDGE], "conf_nn_enabled")
    patch("Sources/CameraView.swift", [VIEW, VIEW_HELP], "conf_nn_enabled")


if __name__ == "__main__":
    main()
