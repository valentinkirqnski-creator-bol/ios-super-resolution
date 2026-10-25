"""One generated sample: a synthetic burst, the app's stages on two alignment variants, labels. Written as one .npz.

Layout (S = margin + patch + margin_hi raw px, P = patch, m = margin, label region = raw [m, m+P] inclusive (P+1 px: the
merge's nearest mask fetch for output pixels [2m, 2m+2P) reads raw pixels m .. m+P)):
  shared:   raw [N,S,S] f16 (noisy, loader domain), clean [N,S,S] f16, gain [N], grey_ref [S,S] f16 (app FFT grey),
            grey [N-1,S,S] f16, cov [N,S/2,S/2,4] f16 (app kernel covariances; frame 0 = reference),
            T [3,2P,2P] f16 (HR target), vis [N-1,P+1,P+1] f16, disp [N-1,2,P+1,P+1] f32 (exact displacement),
            mats [N,L,3,3] f64 (every layer homography), meta (json)
  per variant v in {0, 1}:
            flow{v} [N-1,ny,nx,2] f32, flowal{v}, H{v} [N-1,9] f32, hash{v} [N-1], r0{v} [N-1,S/2,S/2] f16 (app
            analytic mask; constant per RGGB quad, so stored per quad), e{v} [N-1,P+1,P+1] f16 (|q_used - q_gt| raw px), R{v} [N-1,P+1,P+1] f16 (merge
            target), D{v} (val/test: include/exclude utility), cost{v} [P+1,P+1] (cost of R*), kinds{v} (json)
"""
import hashlib
import json
import time

import numpy as np
import torch

from . import corrupt, formation, harness, labels, merge as mg, scenarios, scenes


def seed_of(base, split, idx):
    h = hashlib.sha1(f"{base}/{split}/{idx}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def var_from_clean(clean, gain, prewhite, cfa, a, b):
    """Per-pixel noise variance in the loader domain from the stored clean frames (inverse of formation.render)."""
    N, S, _ = clean.shape
    pw = torch.empty(S, S)
    for i in range(2):
        for j in range(2):
            pw[i::2, j::2] = float(prewhite[cfa[i][j]])
    out = []
    for k in range(N):
        g = float(gain[k])
        s = clean[k] / (pw * g)
        v = (pw * g) ** 2 * (a * s.clamp(0, 1) + b)
        sat = (s >= 1.0 - 1e-4) | ((g > 1.0) & (clean[k] >= pw - 1e-4))
        out.append(torch.where(sat, torch.zeros_like(v), v))
    return torch.stack(out)


def merge_parts(raw, clean, var, cov, hv, scale, cfa, m, P, tile, dense=None, gt_dense=None, vis=None):
    """Merge contributions of every frame over the output block [2m, 2m+2P)^2 (scale 2) for one alignment variant.
    hv: dict(flow [K,ny,nx,2], H [K,9], has_h [K]). dense: optional list of (my, mx) per comparison frame replacing
    the used correspondence. gt_dense + vis [K,P+1,P+1]: also build the ideal counterpart (true correspondences,
    occluded samples dropped) that the split reconstruction cost compares against."""
    o0, Ho = int(round(scale * m)), int(round(scale * P))
    covs = [torch.as_tensor(c, dtype=torch.float32) for c in cov]
    Aref, Acref, Bref, Vref = mg.ref_contrib(raw[0], clean[0], var[0], covs[0], scale, cfa, o0, o0, Ho, Ho)
    A, Ac, B, V = [], [], [], []
    for k in range(1, raw.shape[0]):
        d = None if dense is None else dense[k - 1]
        a_, ac_, b_, v_ = mg.comp_contrib(raw[k], clean[k], var[k], torch.as_tensor(hv["flow"][k - 1]),
                                          torch.as_tensor(hv["H"][k - 1]), bool(hv["has_h"][k - 1] > 0.5), covs[k],
                                          tile, scale, cfa, o0, o0, Ho, Ho, dense=d)
        A.append(a_), Ac.append(ac_), B.append(b_), V.append(v_)
    fy, fx = mg.fetch_index(o0, o0, Ho, Ho, scale, m, m, P + 1, P + 1)
    Ag = Bg = visf = None
    if gt_dense is not None:                   # ideal counterpart: true correspondences, occluded samples dropped
        Ag, Bg = [], []
        zero = torch.zeros_like(raw[0])
        for k in range(1, raw.shape[0]):
            _, ac_, b_, _ = mg.comp_contrib(clean[k], clean[k], zero, torch.as_tensor(hv["flow"][k - 1]),
                                            torch.as_tensor(hv["H"][k - 1]), False, covs[k], tile, scale, cfa,
                                            o0, o0, Ho, Ho, dense=gt_dense[k - 1])
            Ag.append(ac_), Bg.append(b_)
        Ag, Bg = torch.stack(Ag), torch.stack(Bg)
        visf = torch.as_tensor(vis, dtype=torch.float32)[:, fy, fx].unsqueeze(1)
    return mg.Parts(Aref, Acref, Bref, Vref, torch.stack(A), torch.stack(Ac), torch.stack(B), torch.stack(V), fy, fx,
                    Ag, Bg, visf)


def dense_gt(disp, m, P, scale):
    """Ground-truth moved positions for the output block from the exact displacement on raw [m, m+P] (bilinear for
    the half-pixel output positions). disp [K,2,P+1,P+1] float64 -> list of (my, mx) float32 [2P,2P]."""
    Ho = int(round(scale * P))
    lr_y, lr_x = mg.out_grid(int(round(scale * m)), int(round(scale * m)), Ho, Ho, scale)
    ly, lx = lr_y.double() - m, lr_x.double() - m
    y0, x0 = ly.floor().long().clamp(0, P - 1), lx.floor().long().clamp(0, P - 1)
    fy, fx = ly - y0, lx - x0
    out = []
    for k in range(disp.shape[0]):
        d = torch.as_tensor(disp[k], dtype=torch.float64)
        def bil(c):
            return ((1 - fy) * ((1 - fx) * c[y0, x0] + fx * c[y0, x0 + 1]) + fy * ((1 - fx) * c[y0 + 1, x0] + fx * c[y0 + 1, x0 + 1]))
        out.append(((lr_y.double() + bil(d[1])).float(), (lr_x.double() + bil(d[0])).float()))
    return out


def generate(split, idx, cfg, out_path, category=None, keep_utility=None, timing=False):
    t0 = time.time()
    tm = {}
    seed = seed_of(cfg["seed"], split, idx)
    rng = np.random.default_rng(seed)
    if category is None:
        category = (scenarios.CATEGORIES[idx % len(scenarios.CATEGORIES)] if split == "test"
                    else scenarios.pick(rng, cfg["categories"]))
    sp = scenarios.make(category, rng, cfg)
    P, m = int(cfg["patch"]), int(cfg["margin"])
    S = P + m + int(cfg.get("margin_hi", m))
    scale = float(cfg["scale"])
    name = str(rng.choice(cfg["scenes"][split]))
    rgb, cfa, wb, a0, b0 = scenes.load_real(name)
    sc = sp["sc"]
    sc.update(noise_a=a0 * sp["iso"], noise_b=b0 * sp["iso"] ** 2)
    if sp["scene_kind"] == "real":
        src, tex, flip = rgb, float(sp["tex"]), True
    else:
        n_tex = int(sp["tex"] * (S + 2 * 96)) + 64
        src, tex, flip = scenes.procedural(sp["scene_kind"], rng, n_tex, sp["tex"], wb), float(sp["tex"]), False
    ap = cfg["aperture"]
    form = formation.Formation(sc, rng, S, tex, src, cfa, wb, ap["n"], ap["fill"],
                               obj_src=rgb if category == "moving" else None, allow_flip=flip)
    N = form.N
    E = np.asarray(form.b.E, np.float64)
    gains = np.ones(N, np.float32)
    for k in range(1, N):
        if abs(E[k] - E[0]) > 1e-9 * E[0]:
            gains[k] = float(E[0] / E[k] * (1.0 + rng.normal(0, sp["exp_residual"])))
    fr = form.render(int(rng.integers(0, 2 ** 31)), gains)
    tm["render"] = time.time() - t0
    disp_full, vis_full = form.correspondence_grid(0, 0, S, S)
    T = form.hr_target(int(round(scale * m)), int(round(scale * m)), int(round(scale * P)), int(round(scale * P)), scale)
    tm["gt"] = time.time() - t0
    # labels are computed on exactly what the trainer reads back: float16-rounded frames, variance from the clean
    raw, clean = fr["raw"].half().float(), fr["clean"].half().float()
    var = var_from_clean(clean, fr["gain"], wb, cfa, sc["noise_a"], sc["noise_b"])
    gt = disp_full.permute(0, 2, 3, 1).float().numpy()          # [N,S,S,2]
    flags = {"oracle": harness.FLAG_ORACLE, "homog": harness.FLAG_HOMOG, "aligner": 0}[sp["align0"]]
    ref_grey_clean = (0.25 * (clean[0, 0::2, 0::2] + clean[0, 0::2, 1::2] + clean[0, 1::2, 0::2] + clean[0, 1::2, 1::2])).numpy()
    pert1, hp1, kinds1 = corrupt.corrupt_burst(rng, N, S, ref_grey_clean, sp["p_frame"], sp["kinds"], sp["edge_burst"])
    variants = [(np.zeros((N, S, S, 2), np.float32), np.tile(np.eye(3), (N, 1, 1)), [""] * N), (pert1, hp1, kinds1)]
    out = {}
    hc = cfg["harness"]
    tile = int(hc["tile"])
    tune = ([hc["k_detail"], hc["k_denoise"], hc["D_th"], hc["D_tr"], hc.get("k_stretch", 0.0), hc.get("k_shrink", 0.0)]
            if tile > 0 else None)
    hs = []
    for v, (pert, hp, kinds) in enumerate(variants):
        h = harness.run(fr["raw"].numpy(), gt, pert, hp, cfa, scale, sc["noise_a"], sc["noise_b"], wb, flags=flags,
                        seed=seed + v, tile=tile, tune=tune)
        if tune is None:                                      # auto-tuned: variant 1 reuses variant 0's tuning
            tile, tune = h["tile"], list(h["tune"])
        hs.append((h, kinds))
    tm["harness"] = time.time() - t0
    cov = np.concatenate([hs[0][0]["cov_ref"][None], hs[0][0]["cov"]], 0)
    util = (split != "train") if keep_utility is None else keep_utility
    sig2 = mg.single_frame_sig2(T, wb, sc["noise_a"], sc["noise_b"])
    L = cfg["labels"]
    gtd = dense_gt(disp_full[1:, :, m:m + P + 1, m:m + P + 1].numpy(), m, P, scale)
    vis_l = vis_full[1:, m:m + P + 1, m:m + P + 1]
    for v, (h, kinds) in enumerate(hs):
        hv = dict(flow=h["flow"], H=h["H"], has_h=h["has_h"])
        parts = merge_parts(raw, clean, var, cov, hv, scale, cfa, m, P, h["tile"], gt_dense=gtd, vis=vis_l)
        R, pc, D = labels.oracle_R(parts, T, sig2, N - 1, P + 1, P + 1, gamma=L["gamma"], tv=L["tv"],
                                   iters=L["iters"], lr=L["lr"], utility=util, keep_prior=L.get("keep_prior", 0.02),
                                   mode=L.get("mode", "linear"))
        e = torch.stack([labels.align_error(torch.as_tensor(h["flow"][k]), torch.as_tensor(h["H"][k]),
                                            bool(h["has_h"][k] > 0.5), h["tile"],
                                            disp_full[k + 1][:, m:m + P + 1, m:m + P + 1], m, m, P + 1, P + 1)
                         for k in range(N - 1)])
        out.update({f"flow{v}": h["flow"], f"flowal{v}": h["flow_al"], f"H{v}": h["H"], f"hash{v}": h["has_h"],
                    f"r0{v}": h["r0"][:, ::2, ::2].astype(np.float16),   # constant per RGGB quad (verified): guide res
                    f"e{v}": e.numpy().astype(np.float16),
                    f"R{v}": R.numpy().astype(np.float16), f"cost{v}": pc.numpy().astype(np.float32),
                    f"kinds{v}": json.dumps(kinds)})
        if D is not None:
            out[f"D{v}"] = D.numpy().astype(np.float16)
        tm[f"labels{v}"] = time.time() - t0
    mats, info = form.transforms()
    meta = dict(split=split, idx=idx, seed=seed, category=category, scene=name, scene_kind=sp["scene_kind"],
                align0=sp["align0"], N=N, S=S, P=P, m=m, scale=scale, cfa=cfa, prewhite=[float(x) for x in wb],
                noise_a=float(sc["noise_a"]), noise_b=float(sc["noise_b"]), iso=float(sp["iso"]), tile=int(tile),
                tune=[float(x) for x in tune], bits=int(fr["bits"]), world=json.loads(info),
                sc={k: (v if isinstance(v, (int, float, str, bool, list, tuple)) or v is None else str(v))
                    for k, v in sc.items() if k != "motion_table"})
    out.update(raw=raw.numpy().astype(np.float16), clean=clean.numpy().astype(np.float16), gain=fr["gain"],
               grey_ref=hs[0][0]["grey_ref"].astype(np.float16), grey=hs[0][0]["grey"].astype(np.float16),
               cov=cov.astype(np.float16), T=T.numpy().astype(np.float16),
               vis=vis_full[1:, m:m + P + 1, m:m + P + 1].numpy().astype(np.float16),
               disp=disp_full[1:, :, m:m + P + 1, m:m + P + 1].numpy().astype(np.float32), mats=mats,
               meta=json.dumps(meta, default=float))
    tmp = out_path[:-4] + ".tmp.npz"
    np.savez(tmp, **out)
    import os
    os.replace(tmp, out_path)
    tm["total"] = time.time() - t0
    if timing:
        return meta, tm
    return meta


def merge_parts_xy(raw, clean, var, cov, hv, scale, cfa, y0, x0, P, tile, dense=None, gt_dense=None, vis=None,
                   with_noisy=True):
    """merge_parts for an arbitrary crop: output block [s y0, s y0 + s P) x [s x0, s x0 + s P), R map over raw
    [y0, y0 + P] x [x0, x0 + P]. vis [K,P+1,P+1] on that raw region. with_noisy=False skips the noisy-frame
    contributions (A = Ac then; the training losses only use the noise-free merge and the variance)."""
    oy0, ox0, Ho = int(round(scale * y0)), int(round(scale * x0)), int(round(scale * P))
    covs = [torch.as_tensor(c, dtype=torch.float32) for c in cov]
    if not with_noisy:
        raw = clean
    Aref, Acref, Bref, Vref = mg.ref_contrib(raw[0], clean[0], var[0], covs[0], scale, cfa, oy0, ox0, Ho, Ho)
    A, Ac, B, V, Ag, Bg = [], [], [], [], [], []
    zero = torch.zeros_like(raw[0])
    for k in range(1, raw.shape[0]):
        fl, Hm, hh = torch.as_tensor(hv["flow"][k - 1]), torch.as_tensor(hv["H"][k - 1]), bool(hv["has_h"][k - 1] > 0.5)
        d = None if dense is None else dense[k - 1]
        a_, ac_, b_, v_ = mg.comp_contrib(raw[k], clean[k], var[k], fl, Hm, hh, covs[k], tile, scale, cfa,
                                          oy0, ox0, Ho, Ho, dense=d)
        A.append(a_), Ac.append(ac_), B.append(b_), V.append(v_)
        if gt_dense is not None:
            _, ag_, bg_, _ = mg.comp_contrib(clean[k], clean[k], zero, fl, Hm, False, covs[k], tile, scale, cfa,
                                             oy0, ox0, Ho, Ho, dense=gt_dense[k - 1])
            Ag.append(ag_), Bg.append(bg_)
    fy, fx = mg.fetch_index(oy0, ox0, Ho, Ho, scale, y0, x0, P + 1, P + 1)
    visf = None if vis is None else torch.as_tensor(vis, dtype=torch.float32)[:, fy, fx].unsqueeze(1)
    return mg.Parts(Aref, Acref, Bref, Vref, torch.stack(A), torch.stack(Ac), torch.stack(B), torch.stack(V), fy, fx,
                    torch.stack(Ag) if Ag else None, torch.stack(Bg) if Bg else None, visf)


def dense_gt_xy(disp, y0, x0, P, scale):
    """dense_gt for an arbitrary crop: disp [K,2,P+1,P+1] on raw [y0, y0+P] x [x0, x0+P]."""
    Ho = int(round(scale * P))
    lr_y, lr_x = mg.out_grid(int(round(scale * y0)), int(round(scale * x0)), Ho, Ho, scale)
    ly, lx = lr_y.double() - y0, lr_x.double() - x0
    yi, xi = ly.floor().long().clamp(0, P - 1), lx.floor().long().clamp(0, P - 1)
    fy, fx = ly - yi, lx - xi
    out = []
    for k in range(disp.shape[0]):
        d = torch.as_tensor(disp[k], dtype=torch.float64)

        def bil(c):
            return ((1 - fy) * ((1 - fx) * c[yi, xi] + fx * c[yi, xi + 1]) + fy * ((1 - fx) * c[yi + 1, xi] + fx * c[yi + 1, xi + 1]))
        out.append(((lr_y.double() + bil(d[1])).float(), (lr_x.double() + bil(d[0])).float()))
    return out
