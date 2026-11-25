"""Compare the trained fusion network with the app's analytic robustness on the validation
bursts (spatially separate from training), full 256x256 bursts, metrics on the centre 224.

    python eval.py --val DIR/val --model runs/r1/best.pt [--out report.json]

Methods (all through the same exact merge): nn (fp32), nn16 (network in fp16), analytic (the
app's R), all (every frame equal: no robustness), ref (reference frame only).
Metrics: PSNR / SSIM after per-burst exposure normalisation + sRGB curve; error in noise std
on ghost pixels (any frame's flow > 0.5 px off, or occluded) and clean pixels; edge
preservation (gradient magnitude kept on the target's strongest 10 % edges, and the error
there); effective number of frames.
Also: model size, MACs, PC timing of a 12 MP frame, and peak memory of banded 12 MP inference.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import fusion as fu

C = slice(16, 240)


def srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


def ssim(a, b):
    """mean SSIM over channels, 11x11 Gaussian window sigma 1.5, data range 1. a, b [3, h, w]"""
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    k = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    f = lambda x: F.conv2d(x[None], k, groups=3)[0]
    mu_a, mu_b = f(a), f(b)
    saa = f(a * a) - mu_a ** 2
    sbb = f(b * b) - mu_b ** 2
    sab = f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return s.mean().item()


def grad_mag(x):
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return torch.sqrt(gx[..., :-1, :] ** 2 + gy[..., :, :-1] ** 2 + 1e-12).mean(0)


REF_CAP = True
N_IN = fu.N_IN


def nn_weights(net, x, half=False):
    with torch.no_grad():
        if half:
            l = net.half()(x.half()).float()
            net.float()
        else:
            l = net(x)
    return fu.weights_from_logits(l[None], REF_CAP)[0]   # [N, 1, H, W]


def evaluate(val_files, net):
    rows = {k: [] for k in ("nn", "nn16", "analytic", "all", "ref")}
    for p in val_files:
        bu = fu.Burst(p)
        A, B = fu.merge_terms(bu, 0, 0, bu.H, bu.W)
        Ar, Br = fu.merge_terms(bu, 0, 0, bu.H, bu.W, R_analytic=True)
        x = fu.net_inputs(bu, N_IN)
        one = torch.ones(bu.N, 1, 1, 1)
        w32 = nn_weights(net, x)
        w16 = nn_weights(net, x, half=True)
        Rn = torch.cat([torch.ones(1, *bu.R.shape[1:3]), bu.R[..., 0]])          # analytic weights, ref = 1
        outs = dict(nn=(fu.fuse(A, B, w32), w32[:, 0]), nn16=(fu.fuse(A, B, w16), w16[:, 0]),
                    analytic=(fu.fuse(Ar, Br, one), Rn / Rn.sum(0, keepdim=True)),
                    all=(fu.fuse(A, B, one), torch.full((bu.N, bu.H, bu.W), 1.0 / bu.N)),
                    ref=(A[0] / B[0].clamp(min=1e-12), torch.cat([torch.ones(1, bu.H, bu.W), torch.zeros(bu.N - 1, bu.H, bu.W)])))
        tgt = bu.target.permute(2, 0, 1)
        sig = fu.noise_sigma(tgt, bu.a, bu.b)
        norm = torch.quantile(tgt.mean(0).flatten(), 0.99).clamp(min=1e-3)
        gmax = F.interpolate(bu.ghost.amax(0)[None, None], scale_factor=2, mode="nearest")[0, 0]
        ghost = (gmax > 0.5).float()
        slip = ((gmax > 0.15) & (gmax <= 1.5)).float()          # small slips, no gross errors / occlusion
        gt_g = grad_mag(srgb(tgt / norm)[:, C, C])
        edge = gt_g >= torch.quantile(gt_g.flatten(), 0.9)
        for k, (o, wn) in outs.items():
            e = ((o - tgt) / sig).abs().mean(0)[C, C]
            gm = ghost[C, C]
            so, st = srgb(o / norm)[:, C, C], srgb(tgt / norm)[:, C, C]
            mse = ((so - st) ** 2).mean().item()
            og = grad_mag(so)
            sm = slip[C, C]
            se = sm[:-1, :-1] * edge.float()
            rows[k].append(dict(
                psnr=10 * np.log10(1 / max(mse, 1e-12)), ssim=ssim(so, st),
                err=e.mean().item(),
                err_ghost=(e * gm).sum().item() / max(gm.sum().item(), 1), n_ghost=gm.sum().item(),
                err_clean=(e * (1 - gm)).sum().item() / max((1 - gm).sum().item(), 1),
                psnr_ghost=10 * np.log10(1 / max((((so - st) ** 2).mean(0) * gm).sum().item() / max(gm.sum().item(), 1), 1e-12)) if gm.sum() > 0 else float("nan"),
                edge_keep=(og[edge].mean() / gt_g[edge].mean()).item(),
                edge_err=((og - gt_g).abs()[edge].mean()).item(),
                eff_frames=(1.0 / (wn ** 2).sum(0)).mean().item(),
                err_slip=(e * sm).sum().item(), n_slip=sm.sum().item(),
                sq_slip=(((so - st) ** 2).mean(0) * sm).sum().item(),
                edge_err_slip=((og - gt_g).abs() * se).sum().item(), n_edge_slip=se.sum().item()))
    out = {}
    for k, r in rows.items():
        keys = r[0].keys()
        pooled = ("err_ghost", "n_ghost", "err_slip", "n_slip", "sq_slip", "edge_err_slip", "n_edge_slip")
        agg = {m: float(np.nanmean([x[m] for x in r])) for m in keys if m not in pooled}
        ns = sum(x["n_slip"] for x in r)
        agg["err_slip"] = sum(x["err_slip"] for x in r) / max(ns, 1)
        agg["psnr_slip"] = 10 * np.log10(1 / max(sum(x["sq_slip"] for x in r) / max(ns, 1), 1e-12))
        agg["edge_err_slip"] = sum(x["edge_err_slip"] for x in r) / max(sum(x["n_edge_slip"] for x in r), 1)
        agg["slip_frac"] = ns / (len(r) * 224 * 224)
        ng = sum(x["n_ghost"] for x in r)
        agg["err_ghost"] = sum(x["err_ghost"] * x["n_ghost"] for x in r) / max(ng, 1)
        agg["ghost_frac"] = ng / (len(r) * 224 * 224)
        out[k] = agg
    return out


def mobile_numbers(net):
    """Size, MACs, PC CPU timing of one 12 MP frame, peak memory of banded 12 MP inference."""
    import psutil
    npar = sum(p.numel() for p in net.parameters())
    c = net.stem.out_channels
    nb = len(net.blocks)
    cin = net.stem.in_channels
    mac_px = cin * c * 9 + nb * (c * 9 + c * c) + c
    gh, gw = 1512, 2016                                    # 12 MP raw -> guide (half) resolution
    res = dict(params=npar, bytes_fp16=2 * npar, mac_per_guide_px=mac_px,
               gmac_per_frame_12mp=mac_px * gh * gw / 1e9, gmac_8frames_12mp=8 * mac_px * gh * gw / 1e9)
    # banded inference, 128 guide rows + 16-row halo each side (the receptive field), fp32 on PC
    halo, band = 16, 128
    x = torch.randn(1, cin, gh, gw)                    # one frame's inputs (the device builds them per band)
    proc = psutil.Process()
    torch.set_num_threads(6)
    with torch.no_grad():
        net(x[..., :band + 2 * halo, :])                   # warm up
        base = proc.memory_info().rss
        peak = base
        t0 = time.perf_counter()
        logits = torch.empty(1, 1, gh, gw)
        for y in range(0, gh, band):
            a, b = max(0, y - halo), min(gh, y + band + halo)
            l = net(x[..., a:b, :])
            logits[..., y:min(gh, y + band), :] = l[..., y - a:y - a + min(band, gh - y), :]
            peak = max(peak, proc.memory_info().rss)
        dt = time.perf_counter() - t0
        full = net(x)
    res.update(pc_cpu_fp32_s_per_frame_12mp=dt, banded_matches_full=float((full - logits).abs().max()),
               pc_banded_extra_rss_mb=(peak - base) / 2 ** 20,
               device_band_buffers_mb_fp16=2 * (band + 2 * halo) * gw * c * 2 / 2 ** 20,
               device_logit_map_mb_fp16=gh * gw * 2 / 2 ** 20)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--c", type=int, default=16)
    ap.add_argument("--n_in", type=int, default=fu.N_IN)
    ap.add_argument("--no_ref_cap", action="store_true")
    ap.add_argument("--no_mobile", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(6)
    global REF_CAP, N_IN
    REF_CAP, N_IN = not a.no_ref_cap, a.n_in
    net = fu.FusionNet(c=a.c, cin=a.n_in)
    net.load_state_dict(torch.load(a.model))
    net.eval()
    files = sorted(f for d in a.val for f in glob.glob(os.path.join(d, "*.npz")))
    t0 = time.time()
    r = evaluate(files, net)
    print("validation bursts: %d (%.0fs)" % (len(files), time.time() - t0))
    print("%-9s %7s %7s %8s %9s %9s %10s %9s %8s %7s" % ("method", "PSNR", "SSIM", "err", "err_ghost", "err_clean",
                                                       "PSNR_ghost", "edge_keep", "edge_err", "frames"))
    for k, v in r.items():
        print("%-9s %7.2f %7.4f %8.3f %9.3f %9.3f %10.2f %9.3f %8.4f %7.2f" % (
            k, v["psnr"], v["ssim"], v["err"], v["err_ghost"], v["err_clean"], v["psnr_ghost"], v["edge_keep"],
            v["edge_err"], v["eff_frames"]))
    print("ghost pixels: %.1f%% of the evaluated area" % (100 * r["nn"]["ghost_frac"]))
    print("small slips (0.15-1.5 px): %.1f%% of the area" % (100 * r["nn"]["slip_frac"]))
    print("%-9s %9s %10s %14s" % ("method", "err_slip", "PSNR_slip", "edge_err_slip"))
    for k, v in r.items():
        print("%-9s %9.3f %10.2f %14.4f" % (k, v["err_slip"], v["psnr_slip"], v["edge_err_slip"]))
    if a.no_mobile:
        if a.out:
            json.dump(dict(metrics=r), open(a.out, "w"), indent=1)
        return
    m = mobile_numbers(net)
    for k, v in m.items():
        print("%-32s %s" % (k, ("%.4g" % v) if isinstance(v, float) else v))
    if a.out:
        json.dump(dict(metrics=r, mobile=m), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
