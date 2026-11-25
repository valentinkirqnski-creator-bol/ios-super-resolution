"""Merge a real burst with the fusion network, next to the app's analytic merge.

    python merge_real.py --dump out.bin --frames f0.f32 ... --model best.pt --out DIR [--a A --b B]

--dump: wronski_dump's output for the burst (flow, analytic R, kernels, the app's merged image).
Writes DIR/nn.png, DIR/analytic.png (8-bit sRGB, the same global exposure for both),
DIR/frames_nn.png / frames_analytic.png (effective number of frames per pixel, black 1, white 8),
and DIR/merged_nn.f32 (linear RGB, H x W x 3).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import fusion as fu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import read_out  # noqa: E402


class RealBurst:
    def __init__(self, d, frames, a, b, cfa):
        self.frames = torch.from_numpy(np.stack(frames))
        self.N, self.H, self.W = self.frames.shape
        self.covs = torch.from_numpy(np.ascontiguousarray(d["covs"]))
        self.flow = torch.from_numpy(np.ascontiguousarray(d["flows"]))
        self.R = torch.from_numpy(np.ascontiguousarray(np.clip(d["R"], 0, 1)))
        self.a, self.b, self.tile, self.cfa = a, b, d["tile"], cfa


def srgb8(x, gain):
    x = np.clip(x * gain, 0, 1)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (y * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--a", type=float, default=9.60179e-05)
    ap.add_argument("--b", type=float, default=3.40066e-07)
    ap.add_argument("--tile", type=int, default=512)
    a = ap.parse_args()
    torch.set_num_threads(6)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    d = read_out(a.dump)
    H, W = d["merged"].shape[:2]
    frames = [np.fromfile(p, np.float32).reshape(H, W) for p in a.frames]
    bu = RealBurst(d, frames, a.a, a.b, [[2, 1], [1, 0]])
    print("loaded %d frames %dx%d, flow %s, R %s (%.0fs)" % (bu.N, W, H, tuple(bu.flow.shape), tuple(bu.R.shape), time.time() - t0), flush=True)

    net = fu.FusionNet()
    net.load_state_dict(torch.load(a.model))
    net.eval()
    with torch.no_grad():
        x = fu.net_inputs(bu)
        logits = torch.cat([net(x[f:f + 1]) for f in range(bu.N)])              # [N, 1, H/2, W/2]
        del x
        wts = fu.weights_from_logits(logits[None])[0]                            # [N, 1, H, W]
    print("network weights (%.0fs)" % (time.time() - t0), flush=True)

    out = torch.zeros(3, H, W)
    T = a.tile
    with torch.no_grad():
        for y0 in range(0, H, T):
            for x0 in range(0, W, T):
                h, w = min(T, H - y0), min(T, W - x0)
                A, B = fu.merge_terms(bu, y0, x0, h, w)
                out[:, y0:y0 + h, x0:x0 + w] = fu.fuse(A, B, wts[:, :, y0:y0 + h, x0:x0 + w])
                if y0 == 0 and x0 == 0:                                          # parity on real data
                    Ar, Br = fu.merge_terms(bu, y0, x0, h, w, R_analytic=True)
                    pr = fu.fuse(Ar, Br, torch.ones(bu.N, 1, 1, 1))
                    ref = torch.from_numpy(d["merged"][y0:y0 + h, x0:x0 + w]).permute(2, 0, 1)
                    print("parity vs the app's merge (first tile): mean |diff| %.2e  max %.2e" % (
                        (pr - ref).abs().mean(), (pr - ref).abs().max()), flush=True)
    print("merged (%.0fs)" % (time.time() - t0), flush=True)

    nn = out.permute(1, 2, 0).numpy()
    an = d["merged"]
    nn.astype(np.float32).tofile(os.path.join(a.out, "merged_nn.f32"))
    gain = 1.0 / max(np.quantile(an.mean(-1)[::4, ::4], 0.995), 1e-3)
    Image.fromarray(srgb8(nn, gain)).save(os.path.join(a.out, "nn.png"))
    Image.fromarray(srgb8(an, gain)).save(os.path.join(a.out, "analytic.png"))
    eff_nn = (1.0 / (wts[:, 0] ** 2).sum(0)).numpy()
    R = bu.R[..., 0]
    if R.shape[1] != H:
        R = F.interpolate(R[None], size=(H, W), mode="nearest")[0]
    eff_an = ((1 + R.sum(0)) ** 2 / (1 + (R ** 2).sum(0))).numpy()
    for name, e in (("frames_nn.png", eff_nn), ("frames_analytic.png", eff_an)):
        Image.fromarray(np.clip((e - 1) / (bu.N - 1) * 255 + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(a.out, name))
    print("effective frames: nn mean %.2f, analytic mean %.2f" % (eff_nn.mean(), eff_an.mean()))
    print("mean |nn - analytic| in linear: %.2e (%.2f%% of mean signal)" % (np.abs(nn - an).mean(), 100 * np.abs(nn - an).mean() / an.mean()))
    np.save(os.path.join(a.out, "eff.npy"), np.stack([eff_nn, eff_an]).astype(np.float16))
    print("done (%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
