"""Target protocol on a REAL scene with the APP's own aligner and analytic mask.

Scene: one real DNG (APC_1186), decoded by the app's loader, each RGGB/BGGR quad
-> one camera-native RGB texel (no demosaic, noise averaged). Bursts: exact
camera motion over random crops of it (incl. large random shake), rendered in
the device's raw format (BGGR, pre-white-balanced, the DNG's own noise profile
scaled over ISO). Per sample, build/pipe/tp_real.exe runs the app's aligner
(real errors), optional deliberate tile corruption, the app's analytic mask and
the app's merge, with the used flow and with the exact flow.

Target T = app merge, exact flow, R = 1.  Output O = app merge, used flow, R.
Regions: BAD = the used flow is > 0.1 grey px from the truth (aligner error or
corruption, dilated 1 px); CLEAN = the rest. Loss as target_protocol.py.

    python real_protocol.py gen   --out data/rp_train --n 300 --seed0 0
    python real_protocol.py gen   --out data/rp_val   --n 40  --seed0 7000000 --val
    python real_protocol.py eval  --model runs/v4/model.pt
    python real_protocol.py train --init runs/v4/model.pt --out runs/v5
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rnn import scenarios                                         # noqa: E402
from rnn.burst import Burst                                       # noqa: E402
from rnn.features import compute_features, consensus_features, full_consensus_features, NF, NF_CONS   # noqa: E402
from rnn.common import warp_by_flow                                # noqa: E402
CONSENSUS = os.environ.get("RNN_CONSENSUS", "0") == "1"
FULLCONS = os.environ.get("RNN_FULLCONS", "0") == "1"       # v9: full-resolution consensus inputs
HR_TARGET = os.environ.get("RNN_HR_TARGET", "0") == "1"     # v9: true high-res scene as the target
GEO = os.environ.get("RNN_GEO", "0") == "1"                 # v12: geometric misalignment target (0..1)
OCC_LOOK = os.environ.get("RNN_OCC_LOOK", "0") == "1"       # v14: occluded points judged by visible difference
HARD = os.environ.get("RNN_HARD", "0") == "1"               # v14: edge / rotation / flat flags for loss + val
W_EDGE = float(os.environ.get("RNN_W_EDGE", "2"))           # extra cost of keeping a small-shift (< 1 px) misalignment
W_ROT = float(os.environ.get("RNN_W_ROT", "3"))             # ... a misaligned rotated-content pixel
W_FLAT = float(os.environ.get("RNN_W_FLAT", "1"))           # extra cost of dropping a good flat-area pixel
W_TEX = float(os.environ.get("RNN_W_TEX", "0"))             # v18b: ... a good TEXTURED pixel (edge >= 0.5 sigma)
W_CLEAR = float(os.environ.get("RNN_W_CLEAR", "0"))         # v18b: extra cost of any weight on a clearly misaligned pixel
W_NONEDGE = float(os.environ.get("RNN_W_NONEDGE", "0"))     # v22c: extra cost of dropping a good pixel OFF the edge mask
EDGE_HARD = float(os.environ.get("RNN_EDGE_HARD", "0"))     # v27: visible slips ON outlines -> target 0, extra weight
EDGE_KEEP = float(os.environ.get("RNN_EDGE_KEEP", "0"))     # v27: extra cost of dropping a well-aligned OUTLINE pixel
W_NEAR = float(os.environ.get("RNN_W_NEAR", "0"))           # v31: extra cost of dropping a good pixel NEXT TO a slip
NEAR_R = int(os.environ.get("RNN_NEAR_R", "8"))             # v31: ... within this many guide px of one
W_PERFECT = float(os.environ.get("RNN_W_PERFECT", "0"))     # v32: extra cost of dropping a good pixel of an UNcorrupted sample
SELECT_SAFE = os.environ.get("RNN_SELECT", "") == "safe"    # v14: best = fewest errors with leak not above start


GT_NMERGE = float(os.environ.get("RNN_GT_NMERGE", "1"))   # v17: judge visibility at the MERGED image's noise


def geo_target(clean, flow_used, gt_flow, vis, alpha, beta, k=None, floor=None, occ_look=False, n_merge=None,
               geo_tau=None):
    """Continuous target R* in [0,1] per guide pixel and comparison frame. The ghost a flow error paints is
    measured on the NOISE-FREE comparison frame: content at the used flow vs content at the true flow (both from
    the same frame, so aliasing cancels and aligned-but-aliased areas get 1). Compared with the noise sigma plus a
    small visibility floor; out-of-frame points get 0. Occluded points get 0 too, unless occ_look: then they are
    judged by what merging would show (reference vs comparison content at the used flow), so an occluded wall
    that looks like the wall in front of it is not a misalignment. n_merge > 1 (v17): a ghost counts as visible
    against the noise of the MERGED image (single-frame variance / n_merge), because after merging ~8 frames the
    noise drops ~3x and small per-tile errors that hid in one frame's noise show (low-light tile seams, zipper
    edges)."""
    n_merge = GT_NMERGE if n_merge is None else n_merge
    # v22: RNN_GT_K / RNN_GT_FLOOR tighten what counts as visible (defaults 1.5 sigma of the merged noise, 2% floor)
    k = float(os.environ.get("RNN_GT_K", "1.5")) if k is None else k
    floor = float(os.environ.get("RNN_GT_FLOOR", "0.02")) if floor is None else floor
    K = flow_used.shape[0]
    Ck = clean[1:]
    Ce = warp_by_flow(Ck, flow_used)
    Cg = warp_by_flow(Ck, gt_flow)
    from rnn.common import win_mean as _wm
    D2 = _wm((Ce - Cg) ** 2, 1)
    mu = _wm(clean[0:1], 1).clamp_min(0).expand(K, -1, -1)
    edge_rule = os.environ.get("RNN_GT_EDGE")
    if edge_rule:
        # v22b: the stricter rule only on EDGES (see gt_edge_mask), where even a small slip visibly thickens or
        # shifts the edge. Everywhere else the default rule.
        ke, fe = (float(x) for x in edge_rule.split(",")[:2])
        strong = gt_edge_mask(clean, edge_rule)
        k = torch.where(strong, torch.tensor(ke), torch.tensor(float(k)))[None].expand(K, -1, -1)
        floor = torch.where(strong, torch.tensor(fe), torch.tensor(float(floor)))[None].expand(K, -1, -1)
    s2 = (alpha * mu + beta) / 4.0 / n_merge + (floor * mu) ** 2 + 1e-10
    t = vis * torch.exp(-0.5 * D2 / (k * k * s2))
    geo_tau = os.environ.get("RNN_GT_GEO_EDGE") if geo_tau is None else geo_tau   # v24: per-set override
    if geo_tau and float(geo_tau) > 0:
        # v23: on OUTLINES judge geometry, not per-pixel brightness. The eye follows an edge along its length, so a
        # shift of the edge ACROSS itself is seen well below what one 3x3 window shows above noise (the car-roofline
        # artifact). With the true motion known: e_n = |(used - true flow) . edge normal| in raw px; an outline pixel
        # is misaligned in proportion to exp(-0.5 (e_n / tau)^2), tau in raw px.
        tau = float(geo_tau)
        c0 = clean[0]
        gx0, gy0 = torch.zeros_like(c0), torch.zeros_like(c0)
        gx0[:, 1:-1] = 0.5 * (c0[:, 2:] - c0[:, :-2])
        gy0[1:-1] = 0.5 * (c0[2:] - c0[:-2])
        jxx, jyy, jxy = (_wm(v[None], 2)[0] for v in (gx0 * gx0, gy0 * gy0, gx0 * gy0))
        l1 = 0.5 * (jxx + jyy + ((jxx - jyy) ** 2 + 4 * jxy ** 2).sqrt())
        nx, ny = l1 - jyy, jxy                                    # eigenvector of the larger eigenvalue = edge normal
        swap = (nx.abs() + ny.abs()) < 1e-12
        nx, ny = torch.where(swap, jxy, nx), torch.where(swap, l1 - jxx, ny)
        nn_ = (nx * nx + ny * ny).sqrt().clamp_min(1e-12)
        nx, ny = nx / nn_, ny / nn_
        df = flow_used - gt_flow
        e_n = 2.0 * (df[:, 0] * nx + df[:, 1] * ny).abs()       # guide px -> raw px
        if os.environ.get("RNN_GT_GEO_ALL") == "1":
            # v29: the geometry rule on ALL structure that shows after merging, in EVERY set (one consistent label
            # for the same slip). Along a coherent straight edge only the across-edge part shows; on text, texture
            # and corners (low coherence) a slip in any direction shows: e^2 = e_n^2 + ((1 - coh) e_t)^2.
            l2 = 0.5 * (jxx + jyy - ((jxx - jyy) ** 2 + 4 * jxy ** 2).sqrt())
            coh = ((l1 - l2) / (l1 + l2 + 1e-12)).clamp(0, 1)
            e_t = 2.0 * (df[:, 0] * -ny + df[:, 1] * nx).abs()
            e2 = e_n ** 2 + ((1.0 - coh) * e_t) ** 2
            sig_grad = ((alpha * mu[0] + beta) / 8.0 / n_merge).clamp_min(1e-12).sqrt()   # merged-noise gradient sigma
            struct = ((l1.sqrt() > 3.0 * sig_grad)[None].expand(K, -1, -1)) & (vis > 0.5)
            # v32 (RNN_GT_GEO_TAU_TEX): a looser scale on low-coherence texture (foliage, gravel), where a slip of
            # a few tenths of a pixel blurs slightly rather than drawing a visible double edge
            tau_tex = float(os.environ.get("RNN_GT_GEO_TAU_TEX", tau))
            tau_eff = (tau + (tau_tex - tau) * (1.0 - coh))[None]
            t = torch.where(struct, torch.minimum(t, torch.exp(-0.5 * e2 / tau_eff ** 2)), t)
        else:
            outline = gt_edge_mask(clean)[None].expand(K, -1, -1) & (vis > 0.5)
            t = torch.where(outline, torch.minimum(t, torch.exp(-0.5 * (e_n / tau) ** 2)), t)
    if os.environ.get("RNN_GT_CUMUL") == "1" and K > 1:
        # v23: what the frames do TOGETHER. The merge averages the comparison frames, so slips several frames share
        # (the same edge slightly rotated in most of the burst) move the merged edge even when each frame's own slip
        # is below visibility. Compare the visibility-weighted MEAN content at the used vs the true motion; where
        # that merged difference is visible, every frame that contributes to it (own difference >= a quarter of the
        # visibility threshold) is labelled with it too.
        wv = vis.clamp(0, 1)
        nw = wv.sum(0).clamp_min(1e-6)
        Dm2 = _wm((((Ce - Cg) * wv).sum(0) / nw)[None] ** 2, 1)[0]
        tm = torch.exp(-0.5 * Dm2[None] / (k * k * s2))
        contrib = D2 >= (0.5 * k) ** 2 * s2
        t = torch.where(contrib & (vis > 0.5), torch.minimum(t, tm), t)
    if occ_look:
        H, W = Ck.shape[-2:]
        xs = torch.arange(W, dtype=torch.float32).view(1, 1, W) + gt_flow[:, 0]
        ys = torch.arange(H, dtype=torch.float32).view(1, H, 1) + gt_flow[:, 1]
        inside = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
        t_look = torch.exp(-0.5 * _wm((Ce - clean[0:1]) ** 2, 1) / (k * k * s2))
        t = torch.where((vis < 0.5) & inside, t_look, t)
    return t


def gt_edge_mask(clean, edge_rule=None):
    """v22b: where the stricter target applies - a strong, single-direction contrast step in the noise-free
    reference (an outline, roofline, window frame, text stroke; not foliage or texture): local gradient at least
    rel_th of the local brightness per guide px AND structure-tensor coherence at least coh_th, widened 1 px so both
    sides of the edge count. edge_rule = "k,floor,rel_th,coh_th" (RNN_GT_EDGE). Returns a bool [H,W] map."""
    from rnn.common import win_mean as _wm
    edge_rule = edge_rule or os.environ.get("RNN_GT_EDGE", "1.0,0.01,0.08,0.5")
    rel_th, coh_th = (float(x) for x in edge_rule.split(",")[2:4])
    c0 = clean[0]
    mu0 = _wm(clean[0:1], 1).clamp_min(0)[0]
    gx, gy = torch.zeros_like(c0), torch.zeros_like(c0)
    gx[:, 1:-1] = 0.5 * (c0[:, 2:] - c0[:, :-2])
    gy[1:-1] = 0.5 * (c0[2:] - c0[:-2])
    jxx, jyy, jxy = (_wm(v[None], 2)[0] for v in (gx * gx, gy * gy, gx * gy))
    tr, dd = jxx + jyy, ((jxx - jyy) ** 2 + 4 * jxy ** 2).sqrt()
    coh = (dd / (tr + 1e-12)) ** 2
    rel = (gx * gx + gy * gy).sqrt() / mu0.clamp_min(1e-3)
    strong = ((rel >= rel_th) & (coh >= coh_th)).float()
    return F.max_pool2d(strong[None, None], 3, 1, 1)[0, 0] > 0.5


def hard_maps(clean, gt_flow, flow_used, alpha, beta):
    """Flags for the targeted failure modes, per guide pixel: edge = reference |grad| in noise sigmas (3x3 max;
    < 0.5 flat); rot = content whose TRUE motion rotates > ~3 deg relative to the frame (curl of
    the exact flow, motion discontinuities excluded); ferr = used-vs-true flow error in raw px."""
    c0 = clean[0]
    gx, gy = torch.zeros_like(c0), torch.zeros_like(c0)
    gx[:, 1:-1] = 0.5 * (c0[:, 2:] - c0[:, :-2])
    gy[1:-1] = 0.5 * (c0[2:] - c0[:-2])
    sig = ((alpha * c0.clamp_min(0) + beta) / 4.0).sqrt() + 1e-6
    edge = F.max_pool2d(((gx * gx + gy * gy).sqrt() / sig)[None, None], 3, 1, 1)[0, 0]
    fx, fy = gt_flow[:, 0], gt_flow[:, 1]
    curl = torch.zeros_like(fx)
    curl[:, 1:-1, 1:-1] = 0.5 * (fy[:, 1:-1, 2:] - fy[:, 1:-1, :-2]) - 0.5 * (fx[:, 2:, 1:-1] - fx[:, :-2, 1:-1])
    jump = torch.zeros_like(fx)
    for sh in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        jump = torch.maximum(jump, (gt_flow - torch.roll(gt_flow, sh, (2, 3))).norm(dim=1))
    curl = (curl - curl.flatten(1).median(1).values.view(-1, 1, 1)) * (jump < 0.25)
    rot = F.avg_pool2d(curl.abs()[:, None], 5, 1, 2)[:, 0] > 0.1
    return dict(edge=edge, rot=rot, ferr=2.0 * (flow_used - gt_flow).norm(dim=1))
from rnn.model import RobustNet                                   # noqa: E402
from target_protocol import combine, err_map, r_index, region_psnr, SCALE, CLIP   # noqa: E402

TP_REAL = os.environ.get("RNN_TP_REAL", os.path.join(HERE, "build", "pipe", "tp_real.exe"))
REAL = os.path.join(HERE, "data", "real")
TRAIN_SCENES = ["APC_1186"]                             # --scenes overrides; val scene: a different capture
VAL_SCENE = "APC_1211"
SIZE = int(os.environ.get("RNN_SIZE", "384"))         # raw crop per burst (v24: 768 at camera pixel scale)


def load_scene(name):
    """The app loader's output for one DNG -> camera-native RGB per quad, plus its CFA/WB/noise."""
    h, w, c00, c01, c10, c11, a, b, w0, w1, w2, _pw = open(os.path.join(REAL, name + ".meta")).read().split()[:12]
    cfa = ((int(c00), int(c01)), (int(c10), int(c11)))
    wb = np.array([float(w0), float(w1), float(w2)], np.float32)
    r = np.fromfile(os.path.join(REAL, name + ".f32"), np.float32).reshape(int(h), int(w))
    r = r[:r.shape[0] // 2 * 2, :r.shape[1] // 2 * 2]
    q = {}
    for i in range(2):
        for j in range(2):
            q.setdefault(cfa[i][j], []).append(r[i::2, j::2])
    rgb = np.stack([np.mean(q[c], 0) / wb[c] for c in range(3)], -1)
    return np.clip(rgb, 0, None).astype(np.float32), cfa, wb, float(a), float(b)


_SCENE = {}


def build_burst(seed, val, scenes):
    """The synthetic burst of a seed, exactly as _gen_one builds it (same rng draws): returns the Burst (whose
    render_rgb gives any frame's noise-free scene at exact raw positions), the rng positioned just after
    construction, and the scene's cfa / wb / noise. Generation env (RNN_ROT_PATCHES, RNN_VAL_FROM_SCENES) must match
    the env the burst was generated with."""
    global _SCENE
    name_s = scenes[seed % len(scenes)] if (not val or os.environ.get("RNN_VAL_FROM_SCENES") == "1") else VAL_SCENE
    if name_s not in _SCENE:
        _SCENE[name_s] = load_scene(name_s)
    sc_rgb, cfa, wb, alpha0, beta0 = _SCENE[name_s]
    rng = np.random.default_rng(seed)
    name = ["tremor", "tremor", "rotation", "zoom"][seed % 4]
    sc = scenarios.sample(rng, name)
    sc.update(n_objects=0, obj_speed=0.0, blur_prob=0.0, exposure_jitter=0.0, n_frames=int(rng.integers(4, 6)),
              exposure=float(rng.uniform(0.5, 1.6)))
    if rng.random() < 0.35:                                  # big random hand shake
        sc.update(tremor=float(rng.uniform(3.0, 8.0)))
    if name == "rotation":
        sc.update(rot_deg=float(rng.uniform(0.2, 2.0)))
    # rotated tile content: 0..7 patches whose content rotates 0..33 deg in some comparison frames
    rlo, rhi, elo, ehi = (int(x) for x in os.environ.get("RNN_ROT_PATCHES", "0,7,0,4").split(","))
    sc.update(n_rot_patches=int(rng.integers(rlo, rhi + 1)), rot_max_deg=33.0,
              n_rot_patches_edge=int(rng.integers(elo, ehi + 1)))      # PLUS patches centred on edges/detail
    if os.environ.get("RNN_ROT_SMALL"):    # v19: small tilt + shift patches on straight edges (see rnn/burst.py)
        slo, shi = (int(x) for x in os.environ["RNN_ROT_SMALL"].split(","))
        sc.update(n_rot_patches_small=int(rng.integers(slo, shi + 1)))
    if os.environ.get("RNN_SMALL_MODE"):   # v22: tilt_lo,tilt_hi (deg),shift_lo,shift_hi (px): translate / rotate / both
        sc.update(small_mode=tuple(float(x) for x in os.environ["RNN_SMALL_MODE"].split(",")))
        if os.environ.get("RNN_SMALL_CONSISTENT") == "1":   # v23: the same slip in ~90% of frames (see rnn/burst.py)
            sc.update(small_consistent=True)
    if os.environ.get("RNN_EDGE_ROT") == "1":
        # v23: the "same edge slightly rotated" artifact - 24-64 px segments on the strongest straight outlines whose
        # content turns 0.5-3 deg (rotation only, or with a tiny shift) about the segment centre, the SAME way in
        # 40-95% of the frames, so the merge draws the edge twice at a slightly different angle.
        sc.update(small_mode=(0.5, 3.0, 0.03, 0.3), small_consistent=True, small_kinds=(1, 2),
                  small_size=(24.0, 64.0), small_frac=(0.4, 0.95))
    if rng.random() < 0.3:
        sc.update(persp=float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-5)))))
    k = float(np.exp(rng.uniform(0, np.log(30))))            # ISO multiplier on the DNG's own profile
    sc.update(noise_a=alpha0 * k, noise_b=beta0 * k * k)
    # v20 (each draws no rng unless set, so older bursts rebuild unchanged):
    # RNN_BIGROT="sigma_max,drift_max,centre_max": the whole camera turns, per-frame angle spread up to sigma_max deg
    # plus a steady drift of up to drift_max deg per frame (frames end up to ~30 deg from the reference, the range
    # the app's homography seed searches), about a centre up to centre_max px off the crop centre.
    if os.environ.get("RNN_BIGROT"):
        smax, dmax, cmax = (float(x) for x in os.environ["RNN_BIGROT"].split(","))
        sc.update(rot_deg=float(rng.uniform(0.3, smax)),
                  rot_drift=float(np.radians(rng.choice([-1, 1]) * rng.uniform(0.5, dmax))),
                  center_dist=float(rng.uniform(0, cmax)))
    # RNN_PERFECT=1: perfectly alignable bursts - camera motion only, no rotated-content patches (no objects
    # anyway). RNN_PERFECT=still: also no rotation, zoom or perspective, i.e. hand-shake translation the per-tile
    # aligner represents exactly without a homography.
    # With RNN_ROT_SMALL the small tilt/shift patches stay (v22: perfectly aligned except those tiny slips).
    if os.environ.get("RNN_PERFECT"):
        sc.update(n_rot_patches=0, n_rot_patches_edge=0)
        if not os.environ.get("RNN_ROT_SMALL"):
            sc.update(n_rot_patches_small=0)
        if os.environ["RNN_PERFECT"] == "still":
            sc.update(rot_deg=0.0, rot_drift=0.0, zoom_rate=0.0, persp=0.0)
    # scene texels (one per RGGB quad of the DNG) per synthetic raw pixel: 2 (default) shrinks the photo 4x, so
    # edges are sharper / structures smaller per pixel than in a real 12 MP frame; 1 = 2x, 0.5 = the camera's own
    # scale (v19, RNN_TEX_SCALE). Draws no rng.
    # v21 (RNN_TEXT=1): aliased fine-text scenes, perfectly alignable - 8 frames of pure sub-pixel hand-shake
    # translation, a light lens blur (0.3-0.55 raw px) so the sensor aliases, low-to-moderate ISO. The scene is a
    # synthetic text image (rnn/text_scene.py) at 4 texels per raw pixel instead of the DNG photo; the DNG still
    # supplies the CFA, white balance and noise profile. Draws no rng unless set.
    text_bg = None
    if os.environ.get("RNN_TEXT") == "1" or os.environ.get("RNN_CHART") or os.environ.get("RNN_MIX") == "1":
        from rnn.text_scene import text_scene
        from rnn.chart_scene import chart_scene
        sc.update(n_frames=8, optics_sigma=float(rng.uniform(0.3, 0.55)), tremor=float(rng.uniform(0.5, 3.0)),
                  rot_deg=0.0, rot_drift=0.0, zoom_rate=0.0, persp=0.0, n_rot_patches=0, n_rot_patches_edge=0,
                  exposure=float(rng.uniform(0.35, 0.8)), highlights=False)
        if not os.environ.get("RNN_ROT_SMALL"):    # v22 keeps the tiny tilt/shift patches when asked for
            sc.update(n_rot_patches_small=0)
        k = float(np.exp(rng.uniform(0, np.log(8))))
        sc.update(noise_a=alpha0 * k, noise_b=beta0 * k * k)
        tt = float(os.environ.get("RNN_TEX_SCALE", "4"))
        n_tex = int(tt * (SIZE + 2 * 96)) + 64
        cap = tuple(float(x) for x in os.environ.get("RNN_TEXT_CAP", "2.5,7").split(","))   # v22: 1.8,9
        if os.environ.get("RNN_CHART"):
            text_bg = chart_scene(rng, n_tex, n_tex, tt, wb, tuple(float(x) for x in os.environ["RNN_CHART"].split(",")))
        elif os.environ.get("RNN_MIX") == "1":   # demo scene: aliased text on top, edge chart + flat background below
            tb = text_scene(rng, n_tex, n_tex, tt, wb, cap_raw=cap)
            cb = chart_scene(rng, n_tex, n_tex, tt, wb, (3.0, 40.0))
            hh = int(n_tex * 0.55)
            text_bg = np.ascontiguousarray(np.concatenate([tb[:hh], cb[hh:]], 0))
        else:
            text_bg = text_scene(rng, n_tex, n_tex, tt, wb, cap_raw=cap)
    if os.environ.get("RNN_SMALL_DISP"):  # v28: "lo,hi,anywhere,pframe" - uniform-displacement slips (rnn/burst.py)
        dlo_, dhi_, anyw_, pf_ = (float(x) for x in os.environ["RNN_SMALL_DISP"].split(","))
        sc.update(small_disp=(dlo_, dhi_), small_anywhere=anyw_, small_pframe=pf_, small_size=(24.0, 96.0))
    if os.environ.get("RNN_NFRAMES"):     # v24: frames per burst (real bursts have ~8); draws no rng
        sc.update(n_frames=int(os.environ["RNN_NFRAMES"]))
    # v24 (RNN_ROLL="lo,hi" deg): a slight camera ROLL per frame (spread lo-hi, no drift, centre near the crop), no
    # zoom or perspective. Aligned with one translation per tile (tp_real oracle), each tile's piece of a long straight
    # edge is then turned slightly and the edge steps at tile borders - the zippery edge. Draws no rng unless set.
    if os.environ.get("RNN_ROLL"):
        rlo_, rhi_ = (float(x) for x in os.environ["RNN_ROLL"].split(","))
        sc.update(rot_deg=float(rng.uniform(rlo_, rhi_)), rot_drift=0.0, zoom_rate=0.0, persp=0.0,
                  center_dist=float(rng.uniform(0, 100)))
    tex = float(os.environ.get("RNN_TEX_SCALE", "2" if text_bg is None else "4"))
    b = Burst(sc, rng, Hr=SIZE, Wr=SIZE, bg_rgb=sc_rgb if text_bg is None else text_bg, cfa=cfa, prewhite=wb,
              tex_scale=tex, allow_flip=text_bg is None)
    return b, rng, name, cfa, wb, alpha0 * k, beta0 * k * k


def _gen_one(args):
    seed, out, val, scenes = args
    path = os.path.join(out, f"rp{seed:08d}.npz")
    if os.path.exists(path):
        return
    torch.set_num_threads(1)
    b, rng, name, cfa, wb, alpha, beta = build_burst(seed, val, scenes)
    d = b.render(out_scale=1, seed=int(rng.integers(1 << 31)))
    # true high-resolution scene at the app's 2x output grid, same domain as the merge output:
    # native = min(scene * E, 1) (sensor clip), then the loader's pre-white-balance gain per colour
    Ho = int(SIZE * SCALE)
    yy, xx = torch.meshgrid(torch.arange(Ho, dtype=torch.float64) / SCALE, torch.arange(Ho, dtype=torch.float64) / SCALE,
                            indexing="ij")
    rgb = b.render_rgb(0, torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)).view(3, Ho, Ho)
    gt_hr = ((rgb * b.E[0]).clamp(0, 1) * torch.from_numpy(wb)[:, None, None]).permute(1, 2, 0)
    # noise-free camera-native grey per frame (undo the pre-white-balance per CFA site, quad mean): the content
    # each frame really shows, used to measure the ghost a flow error paints (geometric misalignment target)
    rc = d["raw_clean"].numpy()
    g = np.empty_like(rc)
    for i in range(2):
        for j in range(2):
            g[:, i::2, j::2] = rc[:, i::2, j::2] / wb[cfa[i][j]]
    clean_grey = 0.25 * (g[:, 0::2, 0::2] + g[:, 0::2, 1::2] + g[:, 1::2, 0::2] + g[:, 1::2, 1::2])
    clean_rgb = np.zeros((g.shape[0], 3) + clean_grey.shape[1:], np.float32)   # camera-native colour per quad
    for i in range(2):
        for j in range(2):
            clean_rgb[:, cfa[i][j]] += g[:, i::2, j::2] * (0.5 if cfa[i][j] == 1 else 1.0)
    np.savez(path, raw=d["raw"].numpy().astype(np.float32), gt_flow=d["gt_flow"].numpy().astype(np.float32),
             gt_hr=gt_hr.numpy().astype(np.float16), clean_grey=clean_grey.astype(np.float32),
             clean_rgb=clean_rgb.astype(np.float16),
             vis=d["vis"].numpy().astype(np.float16),
             alpha=np.float32(alpha), beta=np.float32(beta), scenario=name,
             cfa=np.array(cfa, np.int32), wb=wb)


def cmd_gen(a):
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with Pool(a.workers) as p:
        jobs = [(s, a.out, a.val, a.scenes) for s in range(a.seed0, a.seed0 + a.n)]
        for i, _ in enumerate(p.imap_unordered(_gen_one, jobs)):
            if i % 50 == 0:
                print(f"{i}/{a.n} {time.time() - t0:.0f}s", flush=True)


NOMERGE = os.environ.get("RNN_NOMERGE", "0") == "1"   # mask-only training: the harness skips all merges
USE_SERVER = os.environ.get("RNN_TP_SERVER", "0") == "1"   # one persistent harness process per worker
_SERVER = None


def _server_job(pi, po):
    """Runs one job on this process's persistent harness (started on first use)."""
    global _SERVER
    if _SERVER is None or _SERVER.poll() is not None:
        _SERVER = subprocess.Popen([TP_REAL, "--server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                   bufsize=1)
    _SERVER.stdin.write(f"{pi} {po}\n")
    _SERVER.stdin.flush()
    line = _SERVER.stdout.readline()
    if not line.startswith("OK 0"):
        _SERVER.kill()
        _SERVER = None
        raise RuntimeError(f"harness failed: {line!r}")


def run_tp_real(raw, gt, alpha, beta, seed, corrupt, cfa, wb, nomerge=False, edge_step=False, homog=False,
                oracle=False, small_only=False):
    """homog (v20): the app's global homography path; flow is then the full correspondence (homography composed)
    and rflow the residual tile flow the net's flow features read. oracle (v21): every tile vector is the exact
    flow (perfect alignment for translation-only bursts), before any deliberate corruption."""
    N, H, W = raw.shape
    Hg, Wg = H // 2, W // 2
    Ho, Wo = int(H * SCALE), int(W * SCALE)
    with tempfile.TemporaryDirectory() as d:
        pi, po = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
        with open(pi, "wb") as fh:
            np.array([N, H, W, seed,
                      int(corrupt) | (2 if nomerge else 0) | (8 if edge_step else 0) | (16 if homog else 0)
                      | (32 if oracle else 0) | (64 if small_only else 0),
                      cfa[0][0], cfa[0][1], cfa[1][0], cfa[1][1]],
                     np.int32).tofile(fh)
            np.array([SCALE, alpha, beta, wb[0], wb[1], wb[2], 1.0], np.float32).tofile(fh)
            raw.astype(np.float32).tofile(fh)
            gt.transpose(0, 2, 3, 1).astype(np.float32).tofile(fh)
        if USE_SERVER:
            _server_job(pi, po)
        else:
            subprocess.run([TP_REAL, pi, po], check=True)
        o = np.fromfile(po, np.float32)
    tile = int(o[-1:].view(np.int32)[0])
    o = torch.from_numpy(o[:-1].copy())
    P, G = Ho * Wo * 3, Hg * Wg
    npl = 7 if homog else 5
    if nomerge:
        g = o
        ref_grey = g[:G].view(Hg, Wg)
        per = g[G:].view(N - 1, npl, Hg, Wg)
        out = dict(ref_grey=ref_grey, grey=per[:, 0], r0=per[:, 1], flow=per[:, 2:4], corr=per[:, 4], tile=tile)
        if homog:
            out["rflow"] = per[:, 5:7]
        return out
    Aref, Bref = o[:P].view(Ho, Wo, 3), o[P:2 * P].view(Ho, Wo, 3)
    m = o[2 * P:2 * P + 4 * (N - 1) * P].view(N - 1, 4, Ho, Wo, 3)
    g = o[2 * P + 4 * (N - 1) * P:]
    ref_grey = g[:G].view(Hg, Wg)
    per = g[G:].view(N - 1, npl, Hg, Wg)
    out = dict(Aref=Aref, Bref=Bref, Au=m[:, 0], Bu=m[:, 1], Ag=m[:, 2], Bg=m[:, 3], ref_grey=ref_grey,
               grey=per[:, 0], r0=per[:, 1], flow=per[:, 2:4], corr=per[:, 4], tile=tile)
    if homog:
        out["rflow"] = per[:, 5:7]
    return out


class RP(Dataset):
    def __init__(self, root, train):
        # root may list several folders separated by commas (several training scenes)
        self.files = sorted(f for r in root.split(",") for f in glob.glob(os.path.join(r, "*.npz")))
        self.train = train

    def __len__(self):
        return len(self.files) * (1 if self.train else 2)

    def __getitem__(self, idx):
        # The app pipeline runs in a subprocess; if one call fails (e.g. memory pressure), retry with a
        # different seed instead of letting the DataLoader wait forever.
        last = None
        for attempt in range(4):
            try:
                return self._item(idx, attempt)
            except Exception as e:   # noqa: BLE001
                last = e
                time.sleep(2.0)
        raise last

    def _item(self, idx, attempt):
        fi = idx % len(self.files)
        z = np.load(self.files[fi])
        raw, gt = z["raw"], z["gt_flow"]
        N, H, W = raw.shape
        rng = np.random.default_rng(None if self.train else 99 + idx)
        corrupt = (bool(rng.random() < float(os.environ.get("RNN_CORRUPT_P", "0.5"))) if self.train
                   else bool(idx >= len(self.files)))
        seed = int(rng.integers(1, 1 << 30)) + attempt
        d = run_tp_real(raw, gt, float(z["alpha"]), float(z["beta"]), seed, corrupt, z["cfa"], z["wb"],
                        nomerge=NOMERGE and GEO)
        Hg, Wg = H // 2, W // 2
        if NOMERGE and GEO:            # mask-only path: features, analytic mask, geometric target
            a_g, b_g = float(z["alpha"]) / 4.0, float(z["beta"]) / 4.0
            Ir = d["ref_grey"][None].expand(N - 1, -1, -1).contiguous()
            with torch.no_grad():
                feats, _ = compute_features(Ir, d["grey"], d["flow"], d["r0"], a_g, b_g, CLIP, max(1, d["tile"] // 2))
            clean, gtf = torch.from_numpy(z["clean_grey"]), torch.from_numpy(gt[1:])
            geo = geo_target(clean, d["flow"], gtf, torch.from_numpy(z["vis"][1:].astype(np.float32)),
                             float(z["alpha"]), float(z["beta"]), occ_look=OCC_LOOK)
            out = dict(feats=feats, r0=d["r0"], geo=geo, corrupt=corrupt, scen=str(z["scenario"]))
            if HARD:
                out.update(hard_maps(clean, gtf, d["flow"], float(z["alpha"]), float(z["beta"])))
            return out
        gy, gx = r_index(H, W, Hg, Wg)
        # ground-truth merge: only samples that truly exist (visible, not at a motion discontinuity, where the
        # merge's flow interpolation would mix two motions). Older data without 'vis' falls back to all samples.
        Rgt = torch.ones(N - 1, Hg, Wg)
        if "vis" in z.files:
            gtf = torch.from_numpy(gt[1:])
            jump = torch.zeros(N - 1, Hg, Wg)
            for sh in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                jump = torch.maximum(jump, (gtf - torch.roll(gtf, sh, (2, 3))).norm(dim=1))
            Rgt = torch.from_numpy(z["vis"][1:].astype(np.float32)) * (jump < 0.25).float()
        T = combine(d["Aref"], d["Bref"], d["Ag"], d["Bg"], Rgt, gy, gx)
        a_g, b_g = float(z["alpha"]) / 4.0, float(z["beta"]) / 4.0           # quad-mean grey noise
        Ir = d["ref_grey"][None].expand(N - 1, -1, -1).contiguous()
        fs = max(1, d["tile"] // 2)
        with torch.no_grad():
            feats, J = compute_features(Ir, d["grey"], d["flow"], d["r0"], a_g, b_g, CLIP, fs)
            if CONSENSUS:
                feats = torch.cat([feats, consensus_features(d["ref_grey"], J, d["r0"], a_g, b_g)], 1)
            if FULLCONS:
                feats = torch.cat([feats, full_consensus_features(torch.from_numpy(raw), d["flow"], d["r0"],
                                                                  float(z["alpha"]), float(z["beta"]))], 1)
        if HR_TARGET:
            T = torch.from_numpy(z["gt_hr"].astype(np.float32))
        geo = None
        if GEO:
            geo = geo_target(torch.from_numpy(z["clean_grey"]), d["flow"], torch.from_numpy(gt[1:]),
                             torch.from_numpy(z["vis"][1:].astype(np.float32)), float(z["alpha"]), float(z["beta"]))
        ferr = (d["flow"] - torch.from_numpy(gt[1:])).norm(dim=1)             # used vs exact, grey px
        badg = ((ferr > 0.1) | (d["corr"] > 0.5)).float().amax(0)
        badg = F.max_pool2d(badg[None, None], 3, 1, 1)[0, 0]
        bad = badg[gy][:, gx] > 0
        h = lambda t: t.half()   # halve inter-process transfer; maths stays fp32 (loss_of upcasts)
        return dict(feats=feats, r0=d["r0"], A=h(d["Au"]), B=h(d["Bu"]), Aref=h(d["Aref"]), Bref=h(d["Bref"]), T=h(T),
                    corr=bad, gy=gy, gx=gx, ferr=ferr, scen=str(z["scenario"]), corrupt=corrupt, flowg=d["flow"],
                    geo=geo if geo is not None else torch.zeros(0))


def _first(x):
    return x[0]


def _wi(_):
    torch.set_num_threads(1)


def loss_of(d, R, w_corr=1.0):
    for k in ("A", "B", "Aref", "Bref", "T"):
        d[k] = d[k].float()
    O = combine(d["Aref"], d["Bref"], d["A"], d["B"], R, d["gy"], d["gx"])
    E = err_map(O, d["T"])
    with torch.no_grad():
        z = torch.zeros_like(R)
        e_ref = err_map(combine(d["Aref"], d["Bref"], d["A"], d["B"], z, d["gy"], d["gx"]), d["T"])
        e_one = err_map(combine(d["Aref"], d["Bref"], d["A"], d["B"], 1 + z, d["gy"], d["gx"]), d["T"])
    clean, corr = ~d["corr"], d["corr"]
    lc = E[clean].mean() / e_ref[clean].mean().clamp_min(1e-6) if clean.any() else torch.zeros(())
    lr_ = E[corr].mean() / e_one[corr].mean().clamp_min(1e-6) if corr.any() else torch.zeros(())
    return lc + w_corr * lr_, float(lc.detach()), float(lr_.detach())


def load_net(path):
    """Loads a RobustNet of whatever input width the checkpoint has."""
    sd = torch.load(path, map_location="cpu")
    if "tile_model" in sd:   # v30: tile-aligned wide view
        from rnn.model import RobustNetTile
        net = RobustNetTile(nf=sd["c1.weight"].shape[1], hid=sd["c1.weight"].shape[0], ctx=sd["k1.weight"].shape[0])
    elif "k1.weight" in sd:   # v28: with the wide-view context branch
        from rnn.model import RobustNetCtx
        net = RobustNetCtx(nf=sd["c1.weight"].shape[1], hid=sd["c1.weight"].shape[0], ctx=sd["k1.weight"].shape[0])
    else:
        net = RobustNet(nf=sd["c1.weight"].shape[1], hid=sd["c1.weight"].shape[0])
    net.load_state_dict(sd)
    net.eval()
    return net


def widen(path, extra, dl):
    """old net -> wider input (new channels inferred from the data), new input weights zero: starts as exactly
    the old net."""
    old = load_net(path)
    extra = next(iter(dl))["feats"].shape[1] - old.c1.weight.shape[1]
    net = RobustNet(nf=old.c1.weight.shape[1] + extra, hid=old.c1.weight.shape[0])
    with torch.no_grad():
        net.c1.weight.zero_()
        net.c1.weight[:, :old.c1.weight.shape[1]] = old.c1.weight
        net.c1.bias.copy_(old.c1.bias)
        for n in ("c2", "c3"):
            getattr(net, n).load_state_dict(getattr(old, n).state_dict())
        net.mu[:old.mu.numel()] = old.mu
        net.sd[:old.sd.numel()] = old.sd
        fs = []
        for i, d in enumerate(dl):          # standardisation of the new channels from data
            fs.append(d["feats"][:, -extra:].permute(1, 0, 2, 3).reshape(extra, -1))
            if i >= 5:
                break
        fs = torch.cat(fs, 1)
        net.mu[-extra:] = fs.mean(1)
        net.sd[-extra:] = fs.std(1).clamp_min(1e-3)
    return net


def widen_hidden(net, hid):
    """16 -> hid hidden channels; new channels feed nothing at start, so the function is unchanged."""
    old = net
    new = RobustNet(nf=old.c1.weight.shape[1], hid=hid)
    h0 = old.c1.weight.shape[0]
    with torch.no_grad():
        new.mu.copy_(old.mu); new.sd.copy_(old.sd)
        new.c1.weight.normal_(0, 0.02); new.c1.bias.zero_()
        new.c1.weight[:h0] = old.c1.weight; new.c1.bias[:h0] = old.c1.bias
        new.c2.weight.zero_(); new.c2.bias.zero_()
        new.c2.weight[:h0, :h0] = old.c2.weight; new.c2.bias[:h0] = old.c2.bias
        new.c2.weight[h0:, :].normal_(0, 0.02)
        new.c3.weight.zero_(); new.c3.bias.copy_(old.c3.bias)
        new.c3.weight[:, :h0] = old.c3.weight
    return new


def cmd_eval(a):
    torch.set_num_threads(4)
    nets = {}
    for spec in a.model:
        net = RobustNet()
        net.load_state_dict(torch.load(spec, map_location="cpu"))
        net.eval()
        nets[os.path.basename(os.path.dirname(spec)) or spec] = net
    dl = DataLoader(RP(a.data, False), 1, shuffle=False, num_workers=a.workers, collate_fn=_first,
                    worker_init_fn=_wi)
    rows = {}
    for d in dl:
        ms = {"ones": torch.ones_like(d["r0"]), "app_r0": d["r0"],
              "oracle": (d["ferr"] <= 0.1).float()}
        with torch.no_grad():
            for k, net in nets.items():
                ms[k] = net(d["feats"], d["r0"])[0]
        tag = "corrupted" if d["corrupt"] else "aligner-only"
        for k, R in ms.items():
            _, lc, lr_ = loss_of(d, R)        # upcasts d's fp16 tensors in place
            O = combine(d["Aref"], d["Bref"], d["A"], d["B"], R, d["gy"], d["gx"])
            r = rows.setdefault((tag, k), {"clean": [], "bad": [], "lc": [], "lr": [], "w_bad": [], "w_good": []})
            r["clean"].append(region_psnr(O, d["T"], ~d["corr"]))
            r["bad"].append(region_psnr(O, d["T"], d["corr"]))
            r["lc"].append(lc)
            r["lr"].append(lr_)
            bad_g = d["ferr"] > 0.1
            if bad_g.any():
                r["w_bad"].append(float(R[bad_g].mean()))
            r["w_good"].append(float(R[~bad_g].mean()))
    for tag in ("aligner-only", "corrupted"):
        print(f"\n[{tag}]  PSNR vs perfect-alignment target (dB)   loss terms (clean: 0 = target; bad: 1 = merge all)")
        print(f"{'mask':10s} {'clean':>7s} {'bad':>7s} | {'L_clean':>8s} {'L_bad':>7s} | {'w bad px':>8s} {'w good px':>9s}")
        for (t, k), r in rows.items():
            if t != tag:
                continue
            print(f"{k:10s} {np.nanmean(r['clean']):7.2f} {np.nanmean(r['bad']):7.2f} | {np.mean(r['lc']):8.4f} "
                  f"{np.mean(r['lr']):7.4f} | {np.nanmean(r['w_bad']):8.3f} {np.mean(r['w_good']):9.3f}")


def geo_loss(R, t, w_bad=9.0, d=None):
    """Weighted BCE toward the geometric target: keeping a misaligned sample (t ~ 0) costs up to 1+w_bad times
    more than dropping a good one. No term rewards merging more. With the hard-case flags (d from RNN_HARD),
    keeping a small-shift misalignment (true error < 1 raw px, visible only because of an edge) or a misaligned
    rotated-content pixel costs more still, and dropping a good flat pixel costs a little more."""
    Rc = R.clamp(1e-5, 1 - 1e-5)
    slip = None
    if EDGE_HARD > 0 and d is not None and "gtedge" in d:
        # v27: a visibly slipped outline pixel is pushed all the way to rejection (a partial weight still merges a
        # faint ghost / thicker edge), and keeping it costs EDGE_HARD x w_bad more
        slip = ((d["gtedge"] > 0.5) & (t < 0.5)).float()
        keep = ((d["gtedge"] > 0.5) & (t > 0.8)).float()   # ... while aligned outlines stay protected (EDGE_KEEP)
        t = t * (1.0 - slip)
    if d is not None and "edge" in d:
        h = 1.0 + W_EDGE * (d["ferr"] < 1).float() + W_ROT * d["rot"].float()
        e = d["edge"][None]
        w = (1.0 + w_bad * (1.0 - t) * h + W_FLAT * t * (e < 0.5).float() + W_TEX * t * (e >= 0.5).float()
             + W_CLEAR * w_bad * (t < 0.05).float())
        if W_NONEDGE > 0 and "gtedge" in d:     # v22c: strict only where an edge slip can show
            w = w + W_NONEDGE * t * (1.0 - d["gtedge"].float())
        if slip is not None:
            w = w + EDGE_HARD * w_bad * slip + EDGE_KEEP * t * keep
        if W_NEAR > 0 and "ferr" in d:
            # v31: the hard negatives - well-aligned pixels whose feature windows reach into a slipped tile. The
            # per-pixel features there look misaligned; only the tile-pooled evidence says otherwise.
            bad = (d["ferr"] > 0.2).float()[:, None]
            near = (torch.nn.functional.max_pool2d(bad, 2 * NEAR_R + 1, 1, NEAR_R)[:, 0] > 0.5) & (d["ferr"] <= 0.2)
            w = w + W_NEAR * t * near.float()
        if W_PERFECT > 0 and "perfect" in d:   # v32: the perfectly aligned twin of every scene must be kept
            w = w + W_PERFECT * t * d["perfect"].float()[:, None, None]
    else:
        w = 1.0 + w_bad * (1.0 - t)
    return (w * -(t * torch.log(Rc) + (1 - t) * torch.log(1 - Rc))).mean()


def stretch(R, lo=0.05, hi=0.95):
    """Continuous R with exact 0 and 1 (the kernel's sharpening)."""
    return ((R - lo) / (hi - lo)).clamp(0, 1)


def cmd_train(a):
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    if a.widen:
        dl0 = DataLoader(RP(a.train, True), 1, shuffle=True, num_workers=a.workers, collate_fn=_first,
                         worker_init_fn=_wi, prefetch_factor=1, timeout=900)
        net = widen(a.init, NF_CONS, dl0)
        del dl0
    else:
        net = load_net(a.init)
        if a.hid and a.hid > net.c1.weight.shape[0]:
            net = widen_hidden(net, a.hid)
        net.train()
    if a.zero_head:   # keep the learned feature layers, start exactly at the app's analytic mask
        torch.nn.init.zeros_(net.c3.weight)
        torch.nn.init.zeros_(net.c3.bias)
    dl = DataLoader(RP(a.train, True), 1, shuffle=True, num_workers=a.workers, collate_fn=_first,
                    persistent_workers=True, worker_init_fn=_wi, prefetch_factor=1, timeout=900)
    dv = DataLoader(RP(a.val, False), 1, shuffle=False, num_workers=a.workers, collate_fn=_first,
                    persistent_workers=False, worker_init_fn=_wi, prefetch_factor=1, timeout=900)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))

    safe = {}          # SELECT_SAFE: errors + leak of the last val, and the starting leak (the ceiling)

    @torch.no_grad()
    def val():
        c, r, x = [], [], []
        for d in dv:
            R = net(d["feats"], d["r0"])[0]
            if GEO:        # accuracy vs the ground-truth mask on clear pixels + similarity on all pixels
                t, Rs = d["geo"], stretch(R)
                dec = (t < 0.2) | (t > 0.8)
                ok = ((t < 0.2) & (Rs < 0.5)) | ((t > 0.8) & (Rs > 0.5))
                c.append((float(ok.sum()), float(dec.sum()), float((Rs - t).abs().sum()), float(t.numel()),
                          float(Rs[t < 0.2].sum()), float((t < 0.2).sum()), float(Rs[t > 0.8].sum()), float((t > 0.8).sum())))
                if "edge" in d:   # the three targeted failure modes
                    bad, good, e = t < 0.2, t > 0.8, d["edge"][None].expand_as(t)
                    miss, over = bad & (Rs >= 0.5), good & (Rs <= 0.5)
                    se, ro, fl = bad & (d["ferr"] < 1), bad & d["rot"], good & (e < 0.5)
                    x.append([float(v.sum()) for v in (miss, bad, over, good, miss & se, se, miss & ro, ro, over & fl, fl)])
            else:
                _, lc, lr_ = loss_of(d, R)
                c.append(lc)
                r.append(lr_)
        if GEO:
            a_ = np.array(c).sum(0)
            acc_, sim_ = a_[0] / a_[1], 1 - a_[2] / a_[3]
            print(f"   accuracy vs GT mask {acc_:.4f}  similarity {sim_:.4f}  leak {a_[4] / a_[5]:.3f}  keep {a_[6] / a_[7]:.3f}",
                  flush=True)
            if x:
                b_ = np.array(x).sum(0)
                q = lambda i: b_[i] / max(1.0, b_[i + 1])
                safe.update(err=q(0) + q(2), leak=a_[4] / a_[5])
                print(f"   balanced {100 * (1 - 0.5 * (q(0) + q(2))):.2f}%  missed {100 * q(0):.2f}%  over-rejected "
                      f"{100 * q(2):.2f}%  | small-shift missed {100 * q(4):.2f}%  rotation missed {100 * q(6):.2f}%  "
                      f"flat over-rejected {100 * q(8):.2f}%", flush=True)
            return 1 - acc_, 1 - sim_
        return float(np.mean(c)), float(np.mean(r))

    def score(vc, vr):
        if SELECT_SAFE and "err" in safe:   # fewest errors, but never leakier than where we started
            return safe["err"] if safe["leak"] <= safe["leak0"] + 0.002 else float("inf")
        return vc + vr

    vc, vr = val()
    print(f"init val: clean {vc:.4f} bad {vr:.4f}", flush=True)
    safe["leak0"] = safe.get("leak", 1.0)
    best = score(vc, vr)
    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
    hist = [dict(step=0, clean=vc, bad=vr)]
    step, acc, t0, run = 0, 0, time.time(), None
    opt.zero_grad()
    while step < a.steps:
        for d in dl:
            R = net(d["feats"], d["r0"])[0]
            if GEO:
                loss = geo_loss(R, d["geo"], d=d)
                lc, lr_ = float(loss.detach()), 0.0
            else:
                loss, lc, lr_ = loss_of(d, R, a.w_corr)
            (loss / a.accum).backward()
            run = (lc, lr_) if run is None else (0.98 * run[0] + 0.02 * lc, 0.98 * run[1] + 0.02 * lr_)
            acc += 1
            if acc % a.accum:
                continue
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"step {step} clean {run[0]:.4f} bad {run[1]:.4f} {time.time() - t0:.0f}s", flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                vc, vr = val()
                hist.append(dict(step=step, clean=vc, bad=vr))
                print(f"[val step {step}] clean {vc:.4f} bad {vr:.4f}", flush=True)
                torch.save(net.state_dict(), os.path.join(a.out, f"model_s{step}.pt"))
                if score(vc, vr) < best:
                    best = score(vc, vr)
                    print(f"   -> new best (step {step})", flush=True)
                    torch.save(net.state_dict(), os.path.join(a.out, "model.pt"))
                json.dump(dict(args=vars(a), hist=hist), open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
            if step >= a.steps:
                break
    print("best", best)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=300)
    g.add_argument("--seed0", type=int, default=0)
    g.add_argument("--val", action="store_true")
    g.add_argument("--scenes", nargs="+", default=TRAIN_SCENES)
    g.add_argument("--workers", type=int, default=10)
    e = sp.add_parser("eval")
    e.add_argument("--model", nargs="+", required=True)
    e.add_argument("--data", default=os.path.join(HERE, "data", "rp_val"))
    e.add_argument("--workers", type=int, default=8)
    t = sp.add_parser("train")
    t.add_argument("--init", required=True)
    t.add_argument("--train", default=os.path.join(HERE, "data", "rp_train"))
    t.add_argument("--val", default=os.path.join(HERE, "data", "rp_val"))
    t.add_argument("--out", default=os.path.join(HERE, "runs", "v5"))
    t.add_argument("--steps", type=int, default=1200)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--workers", type=int, default=10)
    t.add_argument("--eval_every", type=int, default=200)
    t.add_argument("--zero_head", action="store_true")
    t.add_argument("--w_corr", type=float, default=1.0)
    t.add_argument("--widen", action="store_true", help="add the consensus input channels (set RNN_CONSENSUS=1)")
    t.add_argument("--hid", type=int, default=0, help="widen the hidden layer to this many channels")
    a = ap.parse_args()
    {"gen": cmd_gen, "eval": cmd_eval, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
