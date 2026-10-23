"""Emulate the app's fused Metal kernel on CPU and compare against PyTorch on real data.

    python app_kernel_test.py [--model runs/v1/model.pt]
Uses the weights in core/robust_nn_weights.h (what the app ships)."""
import argparse, glob, os, subprocess, sys, tempfile
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rnn.align import densify
from rnn.analytic import analytic_r0, NOMINAL
from rnn.common import warp_by_flow
from rnn.features import compute_features
from rnn.model import RobustNet, RobustNetTile

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(HERE, "runs", "v1", "model.pt"))
ap.add_argument("--lo", type=float, default=0.0)
ap.add_argument("--hi", type=float, default=0.0)
a = ap.parse_args()
core = os.path.join(HERE, "..", "..", "core")
MARK = "// robust_nn (Config::robust_nn_enabled)"
_sd = torch.load(a.model, map_location="cpu")
if "tile_model" in _sd:   # v31: tile-aligned wide view, pooled over the flow tile (fs guide px)
    net = RobustNetTile(nf=_sd["c1.weight"].shape[1], hid=_sd["c1.weight"].shape[0], ctx=_sd["k1.weight"].shape[0])
else:
    net = RobustNet(nf=_sd["c1.weight"].shape[1], hid=_sd["c1.weight"].shape[0])
net.load_state_dict(_sd); net.eval()
worst = 0.0
with tempfile.TemporaryDirectory() as d:
    exe = os.path.join(d, "emu.exe")
    # the kernel exactly as it ships: everything after the marker in HHSRKernels.metal
    src = open(os.path.join(core, "HHSRKernels.metal"), encoding="utf-8").read()
    block = os.path.join(d, "block.metal").replace("\\", "/")
    open(block, "w", encoding="utf-8").write(src[src.index(MARK):])
    a.block = block
    subprocess.run(["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-static", "-I", core,
                    f'-DRNN_APP_BLOCK="{a.block}"', os.path.join(HERE, "cpp", "app_kernel_emu.cpp"), "-o", exe],
                   check=True)
    for f in sorted(glob.glob(os.path.join(HERE, "data", os.environ.get("PARITY_DATA", "val_parity"), "*.npz")))[:4]:
        z = np.load(f)
        k, fs = 1, int(z["tile"])
        Ir = torch.from_numpy(z["grey"][0].astype(np.float32))[None]
        Ic = torch.from_numpy(z["grey"][k].astype(np.float32))[None]
        H, W = Ir.shape[-2:]
        # nearest-tile flow, the layout the app's lookup reproduces
        tf = torch.from_numpy(z["flows"][k][:, ::fs, ::fs].copy()).permute(1, 2, 0)
        fl = densify(tf, H, W, fs)[None]
        na, nb, clip = float(z["noise_a"]), float(z["noise_b"]), 0.9
        r0 = analytic_r0(Ir, warp_by_flow(Ic, fl), na, nb, NOMINAL)
        with torch.no_grad():
            feats = compute_features(Ir, Ic, fl, r0, na, nb, clip, fs)[0]
            R, _ = net(feats, r0, min(fs, 32)) if isinstance(net, RobustNetTile) else net(feats, r0)
        inp, out = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
        with open(inp, "wb") as fh:
            np.array([W, H, fs], np.int32).tofile(fh); np.array([na, nb, clip], np.float32).tofile(fh)
            for t in (Ir, Ic, fl[0].permute(1, 2, 0), r0):
                t.numpy().astype(np.float32).tofile(fh)
        subprocess.run([exe, inp, out, str(a.lo), str(a.hi)], check=True)
        if a.hi > a.lo:
            R = ((R - a.lo) / (a.hi - a.lo)).clamp(0, 1)
        Re = np.fromfile(out, np.float32).reshape(H, W)
        dmax = float(np.abs(Re - R[0].numpy()).max())
        worst = max(worst, dmax)
        print(f"{os.path.basename(f)} fs={fs}: max|R_app - R_torch| = {dmax:.2e}  (mean R {Re.mean():.3f})")
print("APP KERNEL", "OK" if worst < 1e-4 else "FAIL", f"{worst:.2e}")
sys.exit(0 if worst < 1e-4 else 1)
