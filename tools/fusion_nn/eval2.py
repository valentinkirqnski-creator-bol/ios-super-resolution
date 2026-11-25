"""Evaluate the 2x DBSR-style fusion network on gen2.py validation bursts (full 256x256 raw ->
512x512 HR, metrics on the centre 448x448) against the app's analytic R, all frames equal and
the reference alone, all through the same exact merge. Then mobile numbers.

    python eval2.py --val DIR/val --model runs/r3/best.pt [--out eval.json]
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
from eval import grad_mag, srgb, ssim

C = slice(32, 480)


def nn_weights(net, bu, half=False):
    gin, grid, xd = fu.inputs_dbsr(bu)
    with torch.no_grad():
        if half:
            lg = net.half()(gin[None].half(), grid[None].half(), xd[None].half()).float()
            net.float()
        else:
            lg = net(gin[None], grid[None], xd[None])
    j = torch.arange(int(bu.scale) * bu.H, dtype=torch.float32)
    c = (j / bu.scale - 0.5) / 2
    return fu.weights_at(lg, c, c, True)[0]                       # [N, 1, 2H, 2W]


def evaluate(files, net):
    rows = {}
    for p in files:
        bu = fu.Burst(p)
        Ho = int(bu.scale) * bu.H
        A, B = fu.merge_terms(bu, 0, 0, Ho, Ho)
        Ar, Br = fu.merge_terms(bu, 0, 0, Ho, Ho, R_analytic=True)
        one = torch.ones(bu.N, 1, 1, 1)
        w32, w16 = nn_weights(net, bu), nn_weights(net, bu, half=True)
        R = F.interpolate(bu.R[..., 0][None], size=(Ho, Ho), mode="nearest")[0]
        Rn = torch.cat([torch.ones(1, Ho, Ho), R])
        outs = dict(nn=(fu.fuse(A, B, w32), w32[:, 0]), nn16=(fu.fuse(A, B, w16), w16[:, 0]),
                    analytic=(fu.fuse(Ar, Br, one), Rn / Rn.sum(0, keepdim=True)),
                    all=(fu.fuse(A, B, one), torch.full((bu.N, Ho, Ho), 1.0 / bu.N)),
                    ref=(A[0] / B[0].clamp(min=1e-12), torch.cat([torch.ones(1, Ho, Ho), torch.zeros(bu.N - 1, Ho, Ho)])))
        tgt = bu.target.permute(2, 0, 1)
        sig = fu.noise_sigma(tgt, bu.a, bu.b)
        norm = torch.quantile(tgt.mean(0)[::2, ::2].flatten(), 0.99).clamp(min=1e-3)
        gmax = F.interpolate(bu.ghost.amax(0)[None, None], size=(Ho, Ho), mode="nearest")[0, 0][C, C]
        slip = ((gmax > 0.15) & (gmax <= 1.5)).float()
        st = srgb(tgt / norm)[:, C, C]
        gt_g = grad_mag(st)
        edge = (gt_g >= torch.quantile(gt_g.flatten()[::3], 0.9)).float()
        se = slip[:-1, :-1] * edge
        for k, (o, wn) in outs.items():
            e = ((o - tgt) / sig).abs().mean(0)[C, C]
            so = srgb(o / norm)[:, C, C]
            sq = ((so - st) ** 2).mean(0)
            og = grad_mag(so)
            ge = (og - gt_g).abs()
            r = rows.setdefault(k, np.zeros(12))
            r += np.array([10 * np.log10(1 / max(sq.mean().item(), 1e-12)), ssim(so, st), e.mean().item(),
                           (e * slip).sum().item(), (sq * slip).sum().item(), slip.sum().item(),
                           (ge * edge).sum().item(), edge.sum().item(), (ge * se).sum().item(), se.sum().item(),
                           (1.0 / (wn ** 2).sum(0)).mean().item(), 1])
    out = {}
    for k, r in rows.items():
        n = r[11]
        out[k] = dict(psnr=r[0] / n, ssim=r[1] / n, err=r[2] / n, err_slip=r[3] / r[5],
                      psnr_slip=10 * np.log10(1 / max(r[4] / r[5], 1e-12)), edge_err=r[6] / r[7],
                      edge_err_slip=r[8] / max(r[9], 1), eff_frames=r[10] / n, slip_frac=r[5] / (n * 448 * 448))
    return out


def mobile(net):
    import psutil
    enc, dec = net.macs_per_px()
    gh, gw = 1512, 2016                                    # 12 MP raw (device) -> guide grid
    res = dict(params=sum(p.numel() for p in net.parameters()), mac_enc_per_px=enc, mac_dec_per_px=dec,
               gmac_8frames_12mp=(8 * enc + 8 * dec) * gh * gw / 1e9)
    # banded inference of one comparison frame (reference + frame), 128 guide rows + halo 24
    band, halo = 128, 24
    ce, c = net.enc[0].out_channels, net.mix.out_channels
    gin = torch.randn(1, 2, 3, gh, gw)
    proc = psutil.Process()
    torch.set_num_threads(6)
    with torch.no_grad():
        yy, xx = torch.meshgrid(torch.arange(band + 2 * halo, dtype=torch.float32), torch.arange(gw, dtype=torch.float32), indexing="ij")
        grid = torch.stack([(xx + 0.5) / gw * 2 - 1, (yy + 0.5) / (band + 2 * halo) * 2 - 1], -1)[None, None].expand(1, 2, -1, -1, -1)
        xd = torch.zeros(1, 2, fu.N_XD, band + 2 * halo, gw)
        net(gin[..., :band + 2 * halo, :], grid, xd)
        base = proc.memory_info().rss
        peak = base
        t0 = time.perf_counter()
        for y in range(0, gh, band):
            a = max(0, min(y - halo, gh - band - 2 * halo))
            net(gin[..., a:a + band + 2 * halo, :], grid, xd)
            peak = max(peak, proc.memory_info().rss)
        dt = time.perf_counter() - t0
    res.update(pc_cpu_fp32_s_per_frame_12mp=dt, pc_banded_extra_rss_mb=(peak - base) / 2 ** 20,
               device_band_mb_fp16=(band + 2 * halo) * gw * (2 * ce + 2 * c) * 2 / 2 ** 20,
               device_weight_map_mb_fp16=gh * gw * 2 / 2 ** 20)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ce", type=int, default=12)
    ap.add_argument("--c", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(6)
    net = fu.FusionNetD(ce=a.ce, c=a.c)
    net.load_state_dict(torch.load(a.model))
    net.eval()
    files = sorted(f for d in a.val for f in glob.glob(os.path.join(d, "*.npz")))
    t0 = time.time()
    r = evaluate(files, net)
    print("validation bursts: %d (%.0fs), 2x, metrics on the centre 448x448 HR px" % (len(files), time.time() - t0))
    print("%-9s %7s %7s %7s %9s %10s %9s %14s %7s" % ("method", "PSNR", "SSIM", "err", "err_slip", "PSNR_slip",
                                                     "edge_err", "edge_err_slip", "frames"))
    for k, v in r.items():
        print("%-9s %7.2f %7.4f %7.3f %9.3f %10.2f %9.4f %14.4f %7.2f" % (
            k, v["psnr"], v["ssim"], v["err"], v["err_slip"], v["psnr_slip"], v["edge_err"], v["edge_err_slip"], v["eff_frames"]))
    print("small-slip pixels (a frame 0.15-1.5 px off): %.1f%%" % (100 * r["nn"]["slip_frac"]))
    m = mobile(net)
    for k, v in m.items():
        print("%-30s %s" % (k, ("%.4g" % v) if isinstance(v, float) else v))
    if a.out:
        json.dump(dict(metrics=r, mobile=m), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
