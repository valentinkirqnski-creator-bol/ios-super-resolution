"""The app's ConfNet Metal kernels, emulated on CPU (exact kernel source from core/HHSRKernels.metal + the host band
loop of metal_gpu.mm cnn_refine_raw_res, cpp/cnn_kernel_emu.cpp), against PyTorch on stored samples, at several band
heights (band edges exercised; NaN-poisoned band buffers catch any halo shortfall).

    python kernel_test.py [--model runs/m9_guideres/best.pt] [--weights core]   default: random per-frame weights
Prints KERNELS OK / FAIL.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np            # noqa: E402
import torch                  # noqa: E402

from mfsr import model as M                     # noqa: E402
from mfsr.inputs import frame_inputs, guide_grey, grey_noise   # noqa: E402
import export_weights as EW                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
MARK = "// robust_nn (Config::robust_nn_enabled)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--bands", default="32,7")
    ap.add_argument("--samples", type=int, default=2)
    a = ap.parse_args()
    torch.manual_seed(1)
    if a.model:
        ck = torch.load(a.model, map_location="cpu", weights_only=False)
        mc = ck["cfg"]["model"]
        net = M.build(mc)
        net.load_state_dict(ck["state"])
    else:
        mc = dict(ch=16, fusion=False, dils=[1, 2, 4, 8], up_ch=0, use_r0=False, refine_r0=False)
        net = M.build(mc)
        with torch.no_grad():
            net.mu_g.copy_(torch.randn_like(net.mu_g) * 0.5)
            net.sd_g.copy_(torch.rand_like(net.sd_g) + 0.5)
            net.head.weight.mul_(5.0)
    if net.fusion or net.up_ch > 0:
        print("the device path runs only the per-frame, guide-resolution form; nothing to test")
        return
    net.eval()
    core = os.path.join(ROOT, "core")
    worst = 0.0
    with tempfile.TemporaryDirectory() as d:
        hdr = os.path.join(d, "w.h").replace("\\", "/")
        EW.export(net, mc, hdr, tag="kernel test")
        src = open(os.path.join(core, "HHSRKernels.metal"), encoding="utf-8").read()
        block = os.path.join(d, "block.metal").replace("\\", "/")
        import integrate_app as IA                 # the ConfNet kernels are no longer in the app: append them
        cnn = IA.METAL_KERNELS[len(IA.METAL_KERNELS_ANCHOR):] if "cnn_inputs" not in src else ""
        open(block, "w", encoding="utf-8").write(src[src.index(MARK):] + '\n#include "conf_nn_shared.h"\n' + cnn)
        exe = os.path.join(d, "emu.exe")
        subprocess.run(["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-static", "-I", os.path.join(HERE, "cpp", "app"), "-I", core,
                        f'-DCNN_WEIGHTS_HEADER="{hdr}"', f'-DRNN_APP_BLOCK="{block}"',
                        os.path.join(HERE, "cpp", "cnn_kernel_emu.cpp"), "-o", exe], check=True)
        files = sorted(glob.glob(os.path.join(HERE, "data", "d1", "val", "*.npz")))[:a.samples]
        for fn in files:
            z = np.load(fn)
            meta = json.loads(str(z["meta"]))
            cfa = tuple(tuple(r) for r in meta["cfa"])
            wb = np.array(meta["prewhite"], np.float32)
            na, nb, tile = meta["noise_a"], meta["noise_b"], meta["tile"]
            for v in (0, 1):
                raw = torch.from_numpy(z["raw"].astype(np.float32))
                flows, Hs, hh = torch.from_numpy(z[f"flow{v}"]), torch.from_numpy(z[f"H{v}"]), z[f"hash{v}"]
                r0g = torch.from_numpy(z[f"r0{v}"].astype(np.float32))
                xg, xr = frame_inputs(raw, torch.from_numpy(z["grey_ref"].astype(np.float32)),
                                      torch.from_numpy(z["grey"].astype(np.float32)), r0g, flows, Hs, hh, tile, na, nb,
                                      wb, bool(mc.get("use_r0") or mc.get("refine_r0")), cfa)
                K = xg.shape[0]
                with torch.no_grad():
                    lg = net(xg[None], xr[None], torch.ones(1, K),
                             r0g.repeat_interleave(2, -2).repeat_interleave(2, -1)[None] if mc.get("refine_r0") else None)[0]
                Rpy = torch.sigmoid(lg[:, 1, ::2, ::2]) if not mc.get("refine_r0") else None
                gg = guide_grey(raw, wb, cfa)
                a_g, b_g = grey_noise(na, nb)
                G = gg.shape[-1]
                for k in range(K):
                    if Rpy is None:
                        from mfsr.model import ConfNet  # noqa: F401
                    rp = Rpy[k]
                    for B in (int(x) for x in a.bands.split(",")):
                        pi, po = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
                        ny, nx = flows.shape[1:3]
                        with open(pi, "wb") as fh:
                            np.array([G, G, tile, ny, nx, int(hh[k] > 0.5), 0], np.int32).tofile(fh)
                            np.array([a_g, b_g] + list(Hs[k].numpy()), np.float32).tofile(fh)
                            for t in (gg[0], gg[1 + k], flows[k], r0g[k]):
                                np.ascontiguousarray(t.numpy(), np.float32).tofile(fh)
                        subprocess.run([exe, pi, po, str(B)], check=True)
                        rc = torch.from_numpy(np.fromfile(po, np.float32).reshape(G, G))
                        e = float((rc - rp).abs().max()) if torch.isfinite(rc).all() else float("inf")
                        worst = max(worst, e)
                print(f"{os.path.basename(fn)} variant {v} (homography {bool((hh > 0.5).any())}): worst |R_metal - "
                      f"R_torch| so far {worst:.2e}", flush=True)
    print("KERNELS OK" if worst < 5e-4 else "KERNELS FAIL", f"(max abs diff {worst:.2e})")


if __name__ == "__main__":
    main()
