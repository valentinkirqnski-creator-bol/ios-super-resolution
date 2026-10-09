"""Build one burst into everything training needs, and (as __main__) report the
reference points per regime so the dataset can be shown to contain a real
rejection signal before anything is trained.
"""
from __future__ import annotations

import os
import sys

import shutil
import struct
import subprocess
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import srsim
import srmerge
import srburst

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..'))


_CURVES = {}


def sqrt_curves(cfg, n_patches=4000):
    """Per-channel sqrt-domain curves, cached: they depend only on the noise
    gain, and the Monte Carlo is 1001 bins x 4000 patches x 9 samples per
    channel -- far too slow to repeat for every burst."""
    key = tuple(round(v, 12) for v in cfg.alpha_ch + cfg.beta_ch)
    hit = _CURVES.get(key)
    if hit is None:
        std, diff = [], []
        for c in range(3):
            sc_, dc_ = srsim.build_noise_curves_sqrt(cfg.alpha_ch[c],
                                                     cfg.beta_ch[c],
                                                     n_patches=n_patches)
            std.append(sc_)
            diff.append(dc_)
        hit = (std, diff)
        _CURVES[key] = hit
    return hit


def psnr(a, b):
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(m, 1e-12))


# ---------------------------------------------------------------------------
# the REAL aligner (tools/sr_gate/align_tool.exe, built from core/align.cpp)
# ---------------------------------------------------------------------------

ALIGN_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'align_tool.exe')


def real_flow(raws, tile_size, grey_method=0):
    """Per-tile flow as THIS pipeline's block matcher actually produces it.

    The alternative -- corrupting the true flow with a chosen noise model --
    trains the mask to recognise errors no aligner makes. What matters is where
    this matcher fails: repeated texture, low contrast, heavy noise, large
    displacement beyond its search range. That distribution is not guessable,
    so it is measured by running the matcher.

    Verified against known answers before use: identical frames give flow
    exactly 0, and a 3 px roll gives dx = 2.984.

    Returns None if the tool is missing, so callers can fall back rather than
    silently train on something else.
    """
    if not os.path.exists(ALIGN_EXE):
        return None
    h, w = raws[0].shape[:2]
    tmp = tempfile.mkdtemp(prefix='srg_align_')
    try:
        fi = os.path.join(tmp, 'i.bin')
        fo = os.path.join(tmp, 'o.bin')
        with open(fi, 'wb') as f:
            f.write(struct.pack('<5i', h, w, int(tile_size), len(raws),
                                int(grey_method)))
            for r in raws:
                f.write(np.ascontiguousarray(r, dtype=np.float32).tobytes())
        r = subprocess.run([ALIGN_EXE, fi, fo], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(fo):
            return None
        with open(fo, 'rb') as f:
            nc, ny, nx = struct.unpack('<3i', f.read(12))
            a = np.frombuffer(f.read(), dtype=np.float32)
        a = a.reshape(nc, ny, nx, 2).astype(np.float32)
        # Frame 0 is the reference and has no flow; keep the list shape the
        # synthesiser uses so callers index it the same way.
        return [np.zeros((ny, nx, 2), np.float32)] + [a[i] for i in range(nc)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# Image-domain inputs for the gate (in ADDITION to the eight robustness
# statistics). The scalar statistics summarise the residual; these carry its
# spatial structure, which is what distinguishes correct alignment plus aliasing
# from a fraction of a pixel of misalignment at an edge. A small translation
# error across an edge produces a signed +/- residual pair; |d| alone reduces
# that to "large d near a large gradient".
IMG_CH = 12   # I_r(3) I_w(3) E(3) |grad I_r| |grad I_w| grad.grad


def _lum_grad(a):
    """(h, w, 3) -> gx, gy of the channel mean."""
    g = a.mean(axis=2)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2])
    gy[1:-1, :] = 0.5 * (g[2:, :] - g[:-2, :])
    return gx, gy


def warp_guide(comp_means, flow, cfg, nch=3):
    """The comparison guide resampled by the ESTIMATED per-tile flow -- the same
    fetch compute_d_sigma makes internally, lifted out so the warped image
    itself can be handed to the network."""
    h, w = comp_means.shape[:2]
    ts = cfg.tile_size
    ty, tx = srsim.tile_index_grids(h, w, ts, nch)
    ty = np.clip(ty, 0, flow.shape[0] - 1)
    tx = np.clip(tx, 0, flow.shape[1] - 1)
    fsc = 0.5 if nch == 3 else 1.0
    fx = flow[..., 0][ty, tx] * fsc
    fy = flow[..., 1][ty, tx] * fsc
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros_like(comp_means)
    for ch in range(comp_means.shape[2]):
        v = srsim.sample_bilinear_or_inf(comp_means[..., ch], yy + fy, xx + fx)
        out[..., ch] = np.nan_to_num(np.where(np.isfinite(v), v, 0.0),
                                     nan=0.0, posinf=0.0, neginf=0.0)
    return out


def image_features(ref_m, comp_m, flow, cfg):
    """-> (IMG_CH, h, w)."""
    warped = warp_guide(comp_m, flow, cfg)
    e = warped - ref_m
    rgx, rgy = _lum_grad(ref_m)
    wgx, wgy = _lum_grad(warped)
    out = np.concatenate([
        np.transpose(ref_m, (2, 0, 1)),
        np.transpose(warped, (2, 0, 1)),
        np.transpose(e, (2, 0, 1)),
        np.sqrt(rgx * rgx + rgy * rgy)[None],
        np.sqrt(wgx * wgx + wgy * wgy)[None],
        (rgx * wgx + rgy * wgy)[None],
    ], axis=0)
    return out.astype(np.float32)


def flow_error(true_flow_n, flow_n, gh, gw, cfg, scale=2):
    """|F_hat - F_true| per MASK pixel, in raw pixels.

    true_flow_n is per pixel at output resolution (2x raw) in raw-pixel units;
    flow_n is the per-tile estimate the pipeline was handed. Mask pixel g maps
    to raw 2g maps to output index 4g, hence the ::(2*scale) subsample.
    """
    tfx, tfy = true_flow_n
    st = 2 * scale
    tx_ = tfx[::st, ::st][:gh, :gw]
    ty_ = tfy[::st, ::st][:gh, :gw]
    ti, tj = srsim.tile_index_grids(gh, gw, cfg.tile_size, 3)
    ti = np.clip(ti, 0, flow_n.shape[0] - 1)
    tj = np.clip(tj, 0, flow_n.shape[1] - 1)
    ex = flow_n[..., 0][ti, tj] - tx_
    ey = flow_n[..., 1][ti, tj] - ty_
    return np.sqrt(ex * ex + ey * ey).astype(np.float32)


def build_burst(scene, spec, rng, tile_size=16, use_real_align=True):
    """-> dict(feat, Rw, A, B, A_ref, B_ref, gt, meta)

    feat  (N-1, NUM_FEATURES, h, w)   the gate input, guide/raw lattice
    Rw    (N-1, h, w)                 the shipping analytic mask, for reference
    A, B  (N-1, 3, 2h, 2w)            the merge, factored so that
    A_ref, B_ref (3, 2h, 2w)            out = (A_ref + sum A_n R_n)/(B_ref + ...)
    gt    (3, 2h, 2w)                 noise-free frames, true flow, R == 1
    """
    b = srburst.synth_burst(scene, spec, rng, tile_size=tile_size)
    h, w = b['h'], b['w']
    N = len(b['raws'])
    # The flow the mask is scored against is the one the REAL matcher produces,
    # not the synthesiser's fabricated estimate. Fall back only if the tool is
    # missing, and say so, rather than quietly training on a different
    # distribution.
    rf = real_flow(b['raws'], tile_size) if use_real_align else None
    if rf is not None and rf[1].shape[:2] == b['flows'][1].shape[:2]:
        b['flows'] = rf
        b['align'] = 'real'
    else:
        b['align'] = 'synthetic'

    cfg = srsim.Cfg(noise_gain=spec.noise_gain, tile_size=tile_size)
    cfg.use_decimated_guide()
    std_c, diff_c = sqrt_curves(cfg)
    cfg.tune_snr(b['raws'][0], std_c[1])
    # The synthesiser laid the flow out on a fixed tile grid, so keep the mask
    # on that grid rather than letting the SNR tune move it underneath.
    cfg.tile_size = tile_size

    # The guide is HALF resolution and three channels, so the mask lattice is
    # h/2 x w/2 and the loss grid is subsampled to match what that can express.
    ST = 2
    gh, gw = h // 2, w // 2
    Hs, Ws = (h * 2) // ST, (w * 2) // ST
    ref_m, ref_v = srsim.local_stats_3x3(srsim.compute_guide_decimate3(b['raws'][0]))
    covs_ref = srmerge.estimate_kernels(b['raws'][0], cfg)
    A_ref, B_ref = srmerge.accumulate_ref_ab(b['raws'][0], covs_ref, cfg, ST)

    A = np.zeros((N - 1, 3, Hs, Ws), np.float32)
    B = np.zeros_like(A)
    Ag = np.zeros_like(A)
    Bg = np.zeros_like(A)
    At = np.zeros_like(A)
    Bt = np.zeros_like(A)
    # Only the shipped channels are stored. srsim can build 12; channels 8-11
    # measured neutral and core/sr_gate_shared.h declares 8.
    NF = 8
    feat = np.zeros((N - 1, NF, gh, gw), np.float32)
    img = np.zeros((N - 1, IMG_CH, gh, gw), np.float32)
    ferr = np.zeros((N - 1, gh, gw), np.float32)
    Rw = np.zeros((N - 1, gh, gw), np.float32)

    for n in range(1, N):
        covs = srmerge.estimate_kernels(b['raws'][n], cfg)
        fx, fy = srmerge.tile_flow_at_output(b['flows'][n], h, w, cfg.tile_size,
                                             ST)
        A[n - 1], B[n - 1] = srmerge.accumulate_comp_ab(b['raws'][n], fx, fy,
                                                        covs, cfg, ST)

        covs_c = srmerge.estimate_kernels(b['raws_clean'][n], cfg)
        tfx, tfy = b['true_flow'][n]
        Ag[n - 1], Bg[n - 1] = srmerge.accumulate_comp_ab(
            b['raws_clean'][n], tfx[::ST, ::ST], tfy[::ST, ::ST], covs_c, cfg, ST)
        # The SAME noisy frame fetched with the TRUE flow. Differencing this
        # against A/B above isolates the flow error exactly: identical pixels,
        # identical noise realisation, identical kernels -- the only thing that
        # changed is where the merge reached for them. That difference IS the
        # artifact this frame contributes, which is what the network predicts.
        At[n - 1], Bt[n - 1] = srmerge.accumulate_comp_ab(
            b['raws'][n], tfx[::ST, ::ST], tfy[::ST, ::ST], covs, cfg, ST)

        gm, gv = srsim.local_stats_3x3(
            srsim.compute_guide_decimate3(b['raws'][n]))
        d_sq, sig_sq, comps = srsim.compute_d_sigma(ref_m, ref_v, gm, b['flows'][n],
                                                    cfg, std_c, diff_c)
        feat[n - 1] = srsim.build_features(d_sq, sig_sq, ref_m, ref_v,
                                           b['flows'][n], cfg)[:NF]
        img[n - 1] = image_features(ref_m, gm, b['flows'][n], cfg)
        ferr[n - 1] = flow_error(b['true_flow'][n], b['flows'][n], gh, gw, cfg)
        Rw[n - 1] = srsim.wronski_robustness(d_sq, sig_sq, b['flows'][n], ref_m, cfg)

    # ---- the per-frame ARTIFACT target (what the network predicts) --------
    #
    # For frame n, the merge is run twice with everything held fixed except the
    # flow, and the outputs differenced:
    #
    #   out_est  = (A_ref + A_n ) / (B_ref + B_n )   estimated flow
    #   out_true = (A_ref + At_n) / (B_ref + Bt_n)   TRUE flow, same noisy frame
    #   artifact = |out_est - out_true|
    #
    # Why this and not |dF|: 0.3 px of error in flat sky contributes nothing
    # while 0.3 px across a hard edge is a visible double. Same displacement,
    # opposite correct answers -- a label built from displacement cannot
    # separate them, and this one does, because the local image content is
    # already inside it. Why this and not the merged-image error: that mixes all
    # seven frames into one number and leaves the network to guess which was at
    # fault.
    #
    # Expressed in units of the reference frame's own noise sigma, so "1" means
    # an artifact the size of the noise and the policy threshold is in units a
    # viewer's eye works in. sigma is from the noise model at the local
    # brightness, independent of any mask, so it cannot be gamed by rejecting.
    art = np.zeros((N - 1, gh, gw), np.float32)
    bri_ref = np.clip(ref_m.mean(axis=2), 0.0, 1.0)
    if cfg.guide_sqrt:
        bri_lat = bri_ref ** 2
    else:
        bri_lat = bri_ref
    sig_ref = np.sqrt(np.maximum(
        srsim.ALPHA_DNG * spec.noise_gain * bri_lat +
        srsim.BETA_DNG * spec.noise_gain ** 2, 1e-12)).astype(np.float32)
    for n in range(N - 1):
        den_e = np.maximum(B_ref + B[n], 1e-8)
        den_t = np.maximum(B_ref + Bt[n], 1e-8)
        d = np.abs((A_ref + A[n]) / den_e - (A_ref + At[n]) / den_t)
        d = d.mean(axis=0)                       # over colour
        # Output lattice -> mask lattice: the mask governs a block, so take the
        # WORST artifact in it. A mean would let one bad pixel hide behind three
        # good ones, which is the failure being chased.
        bh, bw = d.shape[0] // gh, d.shape[1] // gw
        if bh >= 1 and bw >= 1:
            d = d[:gh * bh, :gw * bw].reshape(gh, bh, gw, bw).max(axis=(1, 3))
        art[n] = (d / np.maximum(sig_ref, 1e-8)).astype(np.float32)

    covs_rc = srmerge.estimate_kernels(b['raws_clean'][0], cfg)
    A_rc, B_rc = srmerge.accumulate_ref_ab(b['raws_clean'][0], covs_rc, cfg, ST)
    ones_out = np.ones((N - 1, Hs, Ws), np.float32)
    gt = srmerge.merge_from_ab(A_rc, B_rc, Ag, Bg, ones_out)

    # An edge map on the ground truth, used to weight the loss and to report
    # edge MSE separately -- ghosting and lost detail both live on edges, and a
    # plain mean over a mostly flat image hides them.
    g = gt.mean(axis=0)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2])
    gy[1:-1, :] = 0.5 * (g[2:, :] - g[:-2, :])
    edge = np.sqrt(gx * gx + gy * gy)

    return dict(feat=feat, img=img, ferr=ferr, art=art, align=b['align'],
                Rw=Rw, A=A, B=B, A_ref=A_ref, B_ref=B_ref, gt=gt,
                edge=edge.astype(np.float32), h=h, w=w,
                stride=ST, guide_scale=2,
                tile_size=tile_size, regime=b['regime'],
                sigma_flow=b['sigma_flow'], noise_gain=b['noise_gain'])


def merged(d, R):
    """R on the guide lattice (N-1, h/2, w/2) -> merged RGB output."""
    n = R.shape[0]
    Rout = np.stack([srmerge.sample_r_at_output(R[i], d['h'], d['w'],
                                                d.get('stride', 2),
                                                d.get('guide_scale', 2))
                     for i in range(n)], axis=0)
    return srmerge.merge_from_ab(d['A_ref'], d['B_ref'], d['A'], d['B'], Rout)


def main():
    dngs, _ = srburst.find_dngs(ROOT)
    scene_full = srburst.load_scene(dngs[0])
    Hs = Ws = 384
    y0 = (scene_full.shape[0] - Hs) // 2
    x0 = (scene_full.shape[1] - Ws) // 2
    scene = np.ascontiguousarray(scene_full[y0:y0 + Hs, x0:x0 + Ws])

    names = {0: 'jitter', 1: 'rot/trans-only', 2: 'moving object', 3: 'outliers'}
    print('%-15s %-8s %-8s %-8s %-8s %-8s %-6s' %
          ('regime', 'sig_f', 'Wronski', 'R=1', 'R=0 ref', 'best', 'meanRw'))
    for regime in (0, 1, 2, 3):
        for trial in range(2):
            rng = np.random.default_rng(100 + regime * 10 + trial)
            spec = srburst.BurstSpec(rng, regime=regime)
            d = build_burst(scene, spec, rng)
            ones = np.ones_like(d['Rw'])
            pw = psnr(merged(d, d['Rw']), d['gt'])
            p1 = psnr(merged(d, ones), d['gt'])
            p0 = psnr(merged(d, np.zeros_like(ones)), d['gt'])
            best = max(pw, p1, p0)
            tag = 'Wronski' if best == pw else ('R=1' if best == p1 else 'R=0')
            print('%-15s %-8.2f %-8.2f %-8.2f %-8.2f %-8s %-6.3f' %
                  (names[regime], d['sigma_flow'], pw, p1, p0, tag, d['Rw'].mean()))


if __name__ == '__main__':
    main()
