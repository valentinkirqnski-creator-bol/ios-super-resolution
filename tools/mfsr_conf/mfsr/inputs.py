"""Per-frame network inputs, all noise-normalised (one weight set across ISO), computed from what the app has at the
point the robustness mask is built: the reference and comparison frames, the tile flow (+ homography) the merge will
use, the analytic mask, the DNG noise model, and the full-resolution greys of the aligner.

Guide resolution (one value per RGGB quad, the grid the analytic mask lives on), NG = 18 channels:
   0-13  rnn.features.compute_features (tools/robust_nn), the 14 features core/robust_nn_shared.h computes on Metal
         today (rnn_features_at with RNN_NF 14): logit(R0)/4 (zeroed when the model does not use the classical mask),
         3x3 mean difference, residual energy (5x5, 9x9), gradient energy, anisotropy, least-squares coherent shift
         |delta| and the fraction of the residual it explains, brightness, clip fraction, affine-fit residuals of
         the flow field (centre, rms, gradient, neighbour jump)
   14-17 rnn.pixnet.pixel_inputs: (J - I)/s, (I - mean I)/s, (J - mean J)/s, (mean J - mean I)/s
         I = reference grey, J = comparison grey warped by the merge's correspondence, s = local noise sigma
         Greys and noise model exactly as the app builds them for its Metal NN path (camera-native guide grey,
         noise (a_R + 2 a_G + a_B) / 16): guide_grey / grey_noise
Raw resolution, NR = 3 channels (what decides WHERE inside a quad an edge is, for a per-pixel R):
   0  (Jf - If)/s_f       If = the aligner's full-resolution grey of the reference, Jf = the comparison's
                          full-resolution grey sampled at the merge's correspondence for that raw pixel
   1  (If - mean3 If)/s_f reference structure
   2  |grad If|/s_f       edge strength
"""
import torch
import torch.nn.functional as F

from rnn.common import bilinear_clamp, win_mean
from rnn.features import compute_features
from rnn.pixnet import pixel_inputs

from . import merge as mg

NG, NR = 18, 3


def grey_noise(a, b, prewhite=None):
    """Noise model of the guide grey, exactly as the app's Metal NN path sets it (metal_gpu.mm rnn15_refine_raw_res:
    noise_a = (alpha_R + 2 alpha_G + alpha_B) / 16, same for beta) with one DNG noise profile for all channels."""
    return a / 4.0, b / 4.0


def quad_mean(x):
    return 0.25 * (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] + x[..., 1::2, 0::2] + x[..., 1::2, 1::2])


def guide_grey(raw, prewhite, cfa):
    """The app's robustness guide reduced to grey, exactly as on device: compute_guide (robustness.cpp) un-prewhitens
    each CFA site to camera-native (x wb_G / wb_c), averages per colour, and rnn_grey_from_guide takes
    0.25 R + 0.5 G + 0.25 B -- i.e. the quad mean of the camera-native sites. raw [...,S,S] loader domain."""
    S = raw.shape[-1]
    inv = torch.empty(raw.shape[-2], S)
    for i in range(2):
        for j in range(2):
            inv[i::2, j::2] = 1.0 / float(prewhite[cfa[i][j]])
    return quad_mean(raw * inv)


def guide_flow(flow, Hm, has_h, tile, G):
    """The guide-pixel correspondence exactly as the app's Metal NN kernels compute it (HHSRKernels.metal RnnAppFlow):
    nearest tile vector at the quad's raw centre (2g + 0.5), halved to guide px; with a homography the point
    2 (g + f/2) is mapped through it and halved back. Returns the warp displacement fl (use_h = 1) and the
    residual tile displacement ffl the flow features read (use_h = 0). -> fl [2,G,G], ffl [2,G,G]."""
    g = torch.arange(G, dtype=torch.float32)
    yy, xx = g[:, None].expand(G, G), g[None, :].expand(G, G)
    ry, rx = 2.0 * yy + 0.5, 2.0 * xx + 0.5
    ny, nx = flow.shape[:2]
    t = float(tile)
    py = torch.trunc(ry / t).long().clamp(0, ny - 1)
    px = torch.trunc(rx / t).long().clamp(0, nx - 1)
    fv = flow.reshape(-1, 2)[py * nx + px]
    fx, fy = 0.5 * fv[..., 0], 0.5 * fv[..., 1]
    sx, sy = xx + fx, yy + fy
    ffl = torch.stack([sx - xx, sy - yy])
    if has_h:
        px2, py2 = 2.0 * sx, 2.0 * sy
        wq = Hm[6] * px2 + Hm[7] * py2 + Hm[8]
        iwq = torch.where(wq != 0, 1.0 / torch.where(wq != 0, wq, torch.ones_like(wq)), torch.zeros_like(wq))
        sx = 0.5 * (Hm[0] * px2 + Hm[1] * py2 + Hm[2]) * iwq
        sy = 0.5 * (Hm[3] * px2 + Hm[4] * py2 + Hm[5]) * iwq
    fl = torch.stack([sx - xx, sy - yy])
    return fl, ffl


def frame_inputs(raw, grey_ref, grey, r0g, flows, Hs, has_h, tile, a, b, prewhite, use_r0, cfa=((2, 1), (1, 0))):
    """Full-frame inputs for every comparison frame.
    raw [N,S,S] noisy loader domain; grey_ref [S,S]; grey [K,S,S] (the aligner's full-res greys); r0g [K,S/2,S/2];
    flows [K,ny,nx,2]; Hs [K,9]. Guide greys are built exactly like the app's (guide_grey). -> xg [K,NG,S/2,S/2],
    xr [K,NR,S,S]."""
    N, S, _ = raw.shape
    K = N - 1
    G = S // 2
    a_g, b_g = grey_noise(a, b)
    gg = guide_grey(raw, prewhite, cfa)
    Ir = gg[0]
    Ic = gg[1:]
    fl, ffl = [], []
    for k in range(K):
        f_, ff_ = guide_flow(flows[k], Hs[k], bool(has_h[k] > 0.5), tile, G)
        fl.append(f_), ffl.append(ff_)
    fl, ffl = torch.stack(fl), torch.stack(ffl)
    Irk = Ir[None].expand(K, G, G).contiguous()
    feats, J = compute_features(Irk, Ic, fl, r0g, a_g, b_g, 0.9, max(1, tile // 2), ffl)
    feats = feats[:, :14].clone()          # the 14 the device's shared header computes (f14 is compiled out there)
    if not use_r0:
        feats[:, 0] = 0.0
    pix = pixel_inputs(Irk, J, a_g, b_g)
    xg = torch.cat([feats, pix], 1)
    # raw resolution
    If = grey_ref
    mu = win_mean(If[None], 1)[0]
    sf = (a_g * mu.clamp_min(0.0) + b_g).clamp_min(1e-10).sqrt()
    yy, xx = torch.meshgrid(torch.arange(S, dtype=torch.float32), torch.arange(S, dtype=torch.float32), indexing="ij")
    gy, gx = torch.gradient(If)
    xr = []
    for k in range(K):
        my, mx = mg.tile_correspondence(yy, xx, flows[k], Hs[k], bool(has_h[k] > 0.5), tile)
        Jf = bilinear_clamp(grey[k][None], mx[None], my[None])[0]
        xr.append(torch.stack([(Jf - If) / sf, (If - mu) / sf, torch.sqrt(gx * gx + gy * gy) / sf]))
    xr = torch.stack(xr).clamp(-10.0, 10.0)
    return xg.clamp(-20.0, 20.0), xr
