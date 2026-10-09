// sr_gate_unet -- the artifact-predicting gate (Config::sr_gate_unet_enabled).
//
// WHAT IT PREDICTS
//
// Not a mask. For each comparison frame and each mask pixel it emits
// log1p(artifact / noise_sigma): how much damage merging that frame there would
// do, in units a viewer's eye works in. sr_gate_unet_mask() in
// sr_gate_shared.h then turns that into R through a policy whose tolerance is
// Config::sr_gate_tau.
//
// That split is the point. The network answers "what will happen", which is a
// fact about the burst and does not change when taste changes; tau answers "how
// much of it do I accept", which is a preference and is a single scalar you can
// turn against real photographs without retraining. A network trained to emit R
// bakes the tolerance into 29813 weights, and every adjustment costs a training
// run.
//
// HOW IT WAS SUPERVISED
//
// On the artifact itself, not on displacement. In synthesis the true flow is
// known, so the merge is run twice with everything held fixed except the flow
// and the outputs differenced; what remains is exactly what that frame's
// misalignment did. The alternative -- a label derived from |F_hat - F_true| --
// cannot work, and that is measured rather than argued: at a fixed 0.25-0.35 px
// of flow error the resulting artifact spans 0.01 to 3.3 sigma, a 106x range,
// because 0.3 px in flat sky does nothing and 0.3 px across a hard edge is a
// visible double. One displacement, a hundredfold spread in consequence.
//
// The flow it trains against comes from running THIS pipeline's block matcher
// over the synthesised bursts (tools/sr_gate/align_tool.cpp), not from a noise
// model applied to the truth. Measured, the matcher's error distribution is
// median 3.67 px with a p90 of 49.8 px -- nothing like the hand-chosen ladder
// that preceded it, and it contains the matcher's real failure modes.
//
// The loss is asymmetric (pinball, q = 0.9): under-predicting an artifact means
// merging a ghost, over-predicting means rejecting a frame that was fine, and
// those are not equally bad. The network is therefore fitted to a high quantile
// -- it errs toward "this will be worse than it looks".
//
// SHAPE  (see SRGU_* in sr_gate_shared.h)
//
//   e1  conv 3x3  20 -> 16   ReLU          full resolution
//   e2  conv 3x3  16 -> 16   ReLU          -> skip
//   avg pool 2x2
//   d1  conv 3x3  16 -> 28   ReLU          half resolution
//   b1  conv 3x3  28 -> 28   ReLU  dil 2
//   b2  conv 3x3  28 -> 28   ReLU  dil 4
//   nearest upsample 2x, concatenate the skip
//   u1  conv 3x3  44 -> 16   ReLU          full resolution
//   head 1x1      16 -> 1    LINEAR
//
// 36 mask pixels of receptive field, against the small gate's 13. Camera motion
// makes an alignment error that is smooth over a wide region, so at any single
// location everything looks locally plausible and the evidence is in the
// coherence of the field across it.
//
// Replicate padding throughout, never zeros: a zero border is an absolute
// position code, and this project has already measured a network learning to
// read one and then working only at its training patch size.
#include "sr_gate.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "parallel.h"
#include "sr_gate_unet_weights.h"

namespace hhsr {

namespace {

constexpr int kIn = SRGU_IN;
constexpr int kBase = SRGU_BASE;
constexpr int kMid = SRGU_MID;

// Rows [y0, y0 + rows) of a dilated 3x3 conv, replicate-padded, src laid out
// channel-interleaved with src_y0 as its first row. Mirrors conv_relu_band in
// sr_gate.cpp; kept separate because the channel counts are runtime here.
void conv_band(const f32* src, int src_y0, int src_rows, f32* dst, int dst_y0,
               int rows, int w, int in_ch, int out_ch, int dil,
               const f32* W, const f32* B, bool relu) {
    parallel_rows(rows, 0, [&](int r) {
        const int y = dst_y0 + r;
        f32* o = dst + (size_t)r * (size_t)w * (size_t)out_ch;
        for (int x = 0; x < w; ++x) {
            f32* ov = o + (size_t)x * (size_t)out_ch;
            for (int oc = 0; oc < out_ch; ++oc) ov[oc] = B[oc];
            for (int ky = -1; ky <= 1; ++ky) {
                int sy = y + ky * dil;
                // Replicate against the BAND's valid span; the caller gives the
                // band enough halo that this only clamps at the true image edge.
                sy = std::min(std::max(sy, src_y0), src_y0 + src_rows - 1);
                const f32* row = src + (size_t)(sy - src_y0) * (size_t)w *
                                       (size_t)in_ch;
                for (int kx = -1; kx <= 1; ++kx) {
                    int sx = x + kx * dil;
                    sx = std::min(std::max(sx, 0), w - 1);
                    const f32* iv = row + (size_t)sx * (size_t)in_ch;
                    const int k = (ky + 1) * 3 + (kx + 1);
                    for (int oc = 0; oc < out_ch; ++oc) {
                        const f32* wrow = W + ((size_t)oc * (size_t)in_ch) * 9;
                        f32 s = 0.f;
                        for (int ic = 0; ic < in_ch; ++ic)
                            s += wrow[(size_t)ic * 9 + k] * iv[ic];
                        ov[oc] += s;
                    }
                }
            }
            if (relu)
                for (int oc = 0; oc < out_ch; ++oc)
                    ov[oc] = ov[oc] > 0.f ? ov[oc] : 0.f;
        }
    });
}

}  // namespace

// metal_gpu.mm uploads the blob verbatim and must not include the generated
// header itself -- the same reason sr_gate_weights() exists for the small gate.
const f32* sr_gate_unet_weights(int* n) {
    *n = SRGU_WEIGHTS_N;
    return kSrGateUNetWeights;
}

bool sr_gate_unet_available() {
    return sizeof(kSrGateUNetWeights) / sizeof(f32) == SRGU_WEIGHTS_N;
}

Image build_sr_gate_unet_features(const Image& ref_means, const Image& ref_vars,
                                  const Image& comp_means, const Image& d_sq,
                                  const Image& sigma_sq, const FlowField& flow,
                                  int tile_size, const Config& cfg) {
    // Channels 0..7 are byte-identical to the small gate's, so they come from
    // the same builder rather than a second copy that could drift.
    Image base = build_sr_gate_features(ref_means, ref_vars, d_sq, sigma_sq,
                                        flow, tile_size, cfg);
    if (base.h <= 0 || base.c != SRG_FEATURES) return Image();
    const int h = base.h, w = base.w, nch = ref_means.c;
    if (comp_means.h != h || comp_means.w != w || comp_means.c != nch)
        return Image();

    Image out(h, w, SRGU_IN);
    // The comparison guide resampled by the per-tile flow: the same fetch the
    // d^2 term makes internally, done here so the warped IMAGE reaches the
    // network and not only a scalar summary of its difference.
    const f32 fsc = (nch == 3) ? 0.5f : 1.f;
    parallel_rows(h, 0, [&](int y) {
        for (int x = 0; x < w; ++x) {
            int pidy, pidx;
            if (nch == 3) {
                pidy = (int)((2.f * (f32)y + 0.5f) / (f32)tile_size);
                pidx = (int)((2.f * (f32)x + 0.5f) / (f32)tile_size);
            } else {
                pidy = y / tile_size;
                pidx = x / tile_size;
            }
            pidy = std::min(std::max(pidy, 0), flow.ny - 1);
            pidx = std::min(std::max(pidx, 0), flow.nx - 1);
            const f32 fx = flow.dx(pidy, pidx) * fsc;
            const f32 fy = flow.dy(pidy, pidx) * fsc;

            // Bilinear, zero outside: an out-of-frame fetch contributes nothing
            // real, and the residual it produces against the reference is then
            // the reference itself, which reads as "very different" -- the
            // correct conclusion.
            auto samp = [&](const Image& im, f32 sy, f32 sx, int ch) -> f32 {
                if (!(sy >= 0.f && sy <= (f32)(im.h - 1) &&
                      sx >= 0.f && sx <= (f32)(im.w - 1)))
                    return 0.f;
                const int y0 = (int)sy, x0 = (int)sx;
                const int y1 = std::min(y0 + 1, im.h - 1);
                const int x1 = std::min(x0 + 1, im.w - 1);
                const f32 ty = sy - (f32)y0, tx = sx - (f32)x0;
                const f32 a = im.at(y0, x0, ch) * (1 - tx) + im.at(y0, x1, ch) * tx;
                const f32 b = im.at(y1, x0, ch) * (1 - tx) + im.at(y1, x1, ch) * tx;
                return a * (1 - ty) + b * ty;
            };

            SrGateUNetInputs in{};
            f32 rsum = 0.f, wsum = 0.f;
            for (int c = 0; c < 3; ++c) {
                const int cc = (nch == 3) ? c : 0;
                in.ref[c] = ref_means.at(y, x, cc);
                in.warp[c] = samp(comp_means, (f32)y + fy, (f32)x + fx, cc);
                rsum += in.ref[c];
                wsum += in.warp[c];
            }
            (void)rsum; (void)wsum;
            // Central differences of the CHANNEL MEAN, unscaled -- see the note
            // on SrGateUNetInputs in sr_gate_shared.h.
            const int xl = std::max(0, x - 1), xr = std::min(w - 1, x + 1);
            const int yu = std::max(0, y - 1), yd = std::min(h - 1, y + 1);
            auto rmean = [&](int yy, int xx) {
                f32 s = 0.f;
                for (int c = 0; c < nch; ++c) s += ref_means.at(yy, xx, c);
                return s / (f32)nch;
            };
            auto wmean = [&](int yy, int xx) {
                f32 s = 0.f;
                for (int c = 0; c < nch; ++c)
                    s += samp(comp_means, (f32)yy + fy, (f32)xx + fx, c);
                return s / (f32)nch;
            };
            // Interior only, matching numpy's zero-filled borders in
            // tools/sr_gate/build.py _lum_grad.
            const bool inx = (x > 0 && x < w - 1);
            const bool iny = (y > 0 && y < h - 1);
            in.rgx = inx ? 0.5f * (rmean(y, xr) - rmean(y, xl)) : 0.f;
            in.rgy = iny ? 0.5f * (rmean(yd, x) - rmean(yu, x)) : 0.f;
            in.wgx = inx ? 0.5f * (wmean(y, xr) - wmean(y, xl)) : 0.f;
            in.wgy = iny ? 0.5f * (wmean(yd, x) - wmean(yu, x)) : 0.f;

            f32* o = &out.at(y, x, 0);
            for (int i = 0; i < SRG_FEATURES; ++i) o[i] = base.at(y, x, i);
            sr_gate_unet_image_features(&in, o + SRG_FEATURES);
        }
    });
    return out;
}

Image sr_gate_unet_infer_cpu(const Image& feat, f32 tau, f32 beta) {
    const int h = feat.h, w = feat.w;
    if (h <= 0 || w <= 0 || feat.c != SRGU_IN || !sr_gate_unet_available())
        return Image();

    const f32* W = kSrGateUNetWeights;
    Image out(h, w, 1);

    // Bands, because 20 channels over a 12 MP plane is 244 MB for the input
    // alone. The U-Net is fully convolutional with a bounded receptive field, so
    // a band carrying SRGU_HALO rows of context on each side can run the whole
    // pool -> bottleneck -> upsample locally and emit only its centre; the skip
    // tensor never leaves the band. The halo is rounded up to an even number so
    // the 2x pooling of the band is exact.
    const int band = 128;
    const int halo = ((SRGU_HALO + 1) / 2) * 2;

    for (int y0 = 0; y0 < h; y0 += band) {
        const int y1 = std::min(y0 + band, h);
        const int by0 = std::max(0, y0 - halo);
        const int by1 = std::min(h, y1 + halo);
        const int bh = by1 - by0;
        const size_t rowb = (size_t)w * (size_t)kBase;

        std::vector<f32> e(bh * rowb), s(bh * rowb);
        conv_band(feat.data.data(), 0, h, e.data(), by0, bh, w, kIn, kBase, 1,
                  W + SRGU_OFF_E1W, W + SRGU_OFF_E1B, true);
        conv_band(e.data(), by0, bh, s.data(), by0, bh, w, kBase, kBase, 1,
                  W + SRGU_OFF_E2W, W + SRGU_OFF_E2B, true);

        // avg pool 2x2. The band height is padded up to even by replication so
        // the halving is exact and the upsample lines back up.
        const int ph = (bh + 1) / 2, pw = (w + 1) / 2;
        std::vector<f32> p((size_t)ph * (size_t)pw * kBase, 0.f);
        parallel_rows(ph, 0, [&](int r) {
            for (int x = 0; x < pw; ++x) {
                const int y0a = std::min(2 * r, bh - 1);
                const int y1a = std::min(2 * r + 1, bh - 1);
                const int x0a = std::min(2 * x, w - 1);
                const int x1a = std::min(2 * x + 1, w - 1);
                f32* o = p.data() + ((size_t)r * pw + x) * kBase;
                const f32* a = s.data() + ((size_t)y0a * w + x0a) * kBase;
                const f32* b = s.data() + ((size_t)y0a * w + x1a) * kBase;
                const f32* c = s.data() + ((size_t)y1a * w + x0a) * kBase;
                const f32* d = s.data() + ((size_t)y1a * w + x1a) * kBase;
                for (int i = 0; i < kBase; ++i)
                    o[i] = 0.25f * (a[i] + b[i] + c[i] + d[i]);
            }
        });

        std::vector<f32> m1((size_t)ph * pw * kMid), m2((size_t)ph * pw * kMid);
        conv_band(p.data(), 0, ph, m1.data(), 0, ph, pw, kBase, kMid, 1,
                  W + SRGU_OFF_D1W, W + SRGU_OFF_D1B, true);
        conv_band(m1.data(), 0, ph, m2.data(), 0, ph, pw, kMid, kMid,
                  SRGU_DIL0, W + SRGU_OFF_B1W, W + SRGU_OFF_B1B, true);
        conv_band(m2.data(), 0, ph, m1.data(), 0, ph, pw, kMid, kMid,
                  SRGU_DIL1, W + SRGU_OFF_B2W, W + SRGU_OFF_B2B, true);

        // Nearest upsample and concatenate the skip: [mid | base] per pixel.
        const int cat = kMid + kBase;
        std::vector<f32> u((size_t)bh * (size_t)w * (size_t)cat);
        parallel_rows(bh, 0, [&](int r) {
            const int sr = std::min(r / 2, ph - 1);
            for (int x = 0; x < w; ++x) {
                const int sx = std::min(x / 2, pw - 1);
                f32* o = u.data() + ((size_t)r * w + x) * cat;
                const f32* mi = m1.data() + ((size_t)sr * pw + sx) * kMid;
                const f32* sk = s.data() + ((size_t)r * w + x) * kBase;
                for (int i = 0; i < kMid; ++i) o[i] = mi[i];
                for (int i = 0; i < kBase; ++i) o[kMid + i] = sk[i];
            }
        });

        // Only the rows this band owns are written out.
        const int rows = y1 - y0;
        std::vector<f32> fin((size_t)rows * rowb);
        conv_band(u.data(), by0, bh, fin.data(), y0, rows, w, cat, kBase, 1,
                  W + SRGU_OFF_U1W, W + SRGU_OFF_U1B, true);

        const f32* hw = W + SRGU_OFF_HW;
        const f32 hb = W[SRGU_OFF_HB];
        parallel_rows(rows, 0, [&](int r) {
            const f32* src = fin.data() + (size_t)r * rowb;
            for (int x = 0; x < w; ++x) {
                const f32* v = src + (size_t)x * kBase;
                f32 z = hb;
                for (int i = 0; i < kBase; ++i) z += hw[i] * v[i];
                out.at(y0 + r, x) = sr_gate_unet_mask(z, tau, beta);
            }
        });
    }
    return out;
}

Image sr_gate_unet_mask_image(const Image& ref_means, const Image& ref_vars,
                              const Image& comp_means, const Image& d_sq,
                              const Image& sigma_sq, const FlowField& flow,
                              int tile_size, const Config& cfg) {
    if (!cfg.sr_gate_unet_enabled || !sr_gate_unet_available()) return Image();
    // Same domain restriction as the small gate: the weights are fitted on the
    // three-channel half-resolution guide.
    if (ref_means.c != 3) return Image();
    Image feat = build_sr_gate_unet_features(ref_means, ref_vars, comp_means,
                                             d_sq, sigma_sq, flow, tile_size,
                                             cfg);
    if (feat.h <= 0) return Image();
    return sr_gate_unet_infer_cpu(feat, cfg.sr_gate_tau, cfg.sr_gate_beta);
}

}  // namespace hhsr
