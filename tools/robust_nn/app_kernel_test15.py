"""Emulate the app's v15 banded Metal kernels on CPU (exact kernel source + the host band loop) and compare against
PyTorch's PixNet with the shipped logit shift, on real data, at several band heights (band edges exercised).

    python app_kernel_test15.py [--model runs/v15b/pix_s15000.pt --shift -0.75] [--lo 0.2 --hi 0.9]
Uses the weights in core/robust_nn15_weights.h (what the app ships); --model/--shift must match its export."""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rnn.align import densify                      # noqa: E402
from rnn.analytic import analytic_r0, NOMINAL      # noqa: E402
from rnn.common import warp_by_flow                # noqa: E402
from rnn.features import compute_features          # noqa: E402
from rnn.pixnet import PixNet, pixel_inputs        # noqa: E402
from real_protocol import load_net                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/v15b/model.pt")
ap.add_argument("--base", default="runs/v14/model.pt")
ap.add_argument("--shift", type=float, default=-0.75)
ap.add_argument("--lo", type=float, default=0.0)
ap.add_argument("--hi", type=float, default=0.0)
ap.add_argument("--bands", default="32,7")
a = ap.parse_args()
core = os.path.join(HERE, "..", "..", "core")
MARK = "// robust_nn (Config::robust_nn_enabled)"
sd = torch.load(os.path.join(HERE, a.model), map_location="cpu")
net = PixNet(load_net(os.path.join(HERE, a.base)), ch=sd["p1.weight"].shape[0])
net.load_state_dict(sd)
with torch.no_grad():
    net.p5.bias += a.shift                         # the shipped operating point, exactly as exported
net.eval()
worst = 0.0
with tempfile.TemporaryDirectory() as d:
    exe = os.path.join(d, "emu15.exe")
    src = open(os.path.join(core, "HHSRKernels.metal"), encoding="utf-8").read()
    block = os.path.join(d, "block.metal").replace("\\", "/")
    open(block, "w", encoding="utf-8").write(src[src.index(MARK):])
    subprocess.run(["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-static", "-I", core,
                    f'-DRNN_APP_BLOCK="{block}"', os.path.join(HERE, "cpp", "app_kernel_emu15.cpp"), "-o", exe],
                   check=True)
    files = sorted(glob.glob(os.path.join(HERE, "data", os.environ.get("PARITY_DATA", "val_parity"), "*.npz")))[:3]
    for f in files:
        z = np.load(f)
        k, fs = 1, int(z["tile"])
        Ir = torch.from_numpy(z["grey"][0].astype(np.float32))[None]
        Ic = torch.from_numpy(z["grey"][k].astype(np.float32))[None]
        H, W = Ir.shape[-2:]
        tf = torch.from_numpy(z["flows"][k][:, ::fs, ::fs].copy()).permute(1, 2, 0)
        fl = densify(tf, H, W, fs)[None]
        na, nb, clip = float(z["noise_a"]), float(z["noise_b"]), 0.9
        r0 = analytic_r0(Ir, warp_by_flow(Ic, fl), na, nb, NOMINAL)
        with torch.no_grad():
            feats, J = compute_features(Ir, Ic, fl, r0, na, nb, clip, fs)
            R = net(feats, r0, pixel_inputs(Ir, J, na, nb))[0][0]
        if a.hi > a.lo:
            R = ((R - a.lo) / (a.hi - a.lo)).clamp(0, 1)
        inp = os.path.join(d, "in.bin")
        with open(inp, "wb") as fh:
            np.array([W, H, fs], np.int32).tofile(fh)
            np.array([na, nb, clip], np.float32).tofile(fh)
            for t in (Ir, Ic, fl[0].permute(1, 2, 0), r0):
                t.numpy().astype(np.float32).tofile(fh)
        for band in (int(x) for x in a.bands.split(",")):
            out = os.path.join(d, f"out{band}.bin")
            subprocess.run([exe, inp, out, str(band), str(a.lo), str(a.hi)], check=True)
            Re = np.fromfile(out, np.float32).reshape(H, W)
            nan = int(np.isnan(Re).sum())
            dmax = float(np.nanmax(np.abs(Re - R.numpy()))) if nan < Re.size else float("inf")
            worst = max(worst, dmax if nan == 0 else float("inf"))
            print(f"{os.path.basename(f)} {W}x{H} fs={fs} band={band}: max|R_app - R_torch| = {dmax:.2e}  "
                  f"NaN {nan}  (mean R {np.nanmean(Re):.3f})")
print("APP KERNEL v15", "OK" if worst < 1e-4 else "FAIL", f"{worst:.2e}")
sys.exit(0 if worst < 1e-4 else 1)
