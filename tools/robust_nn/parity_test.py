"""C++ (shared header) vs PyTorch parity on real synthetic data.

    python parity_test.py                       # random-weight net (exercises every weight)
    python parity_test.py --model runs/v1/model.pt
"""
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
from rnn.analytic import analytic_r0, NOMINAL          # noqa: E402
from rnn.common import warp_by_flow                    # noqa: E402
from rnn.features import compute_features, NAMES       # noqa: E402
from rnn.model import RobustNet                        # noqa: E402
from export_weights import write_header                # noqa: E402


def build(header, exe):
    cpp = os.path.join(HERE, "cpp")
    core = os.path.join(HERE, "..", "..", "core")   # canonical home of robust_nn_shared.h
    cmd = ["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-fopenmp", "-static",
           f'-DRNN_WEIGHTS_HEADER="{header}"', "-I", core, "-I", cpp,
           os.path.join(cpp, "robust_nn.cpp"), os.path.join(cpp, "robust_nn_cli.cpp"), "-o", exe]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "val"))
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    torch.manual_seed(1)
    net = RobustNet()
    if a.model:
        net.load_state_dict(torch.load(a.model, map_location="cpu"))
    else:   # random weights everywhere, incl. the zero-initialised head and the standardisation
        for p in net.parameters():
            torch.nn.init.normal_(p, 0, 0.3)
        net.mu.normal_(0, 0.5)
        net.sd.uniform_(0.5, 2.0)
    net.eval()
    worst_f, worst_r = 0.0, 0.0
    with tempfile.TemporaryDirectory() as d:
        hdr = os.path.join(d, "w.h").replace("\\", "/")
        exe = os.path.join(d, "rnn_cli.exe")
        write_header(net, hdr, "parity test")
        build(hdr, exe)
        files = sorted(glob.glob(os.path.join(a.data, "*.npz")))[:a.n]
        for fi, f in enumerate(files):
            z = np.load(f)
            k = 1 + fi % (z["grey"].shape[0] - 1)
            Ir = torch.from_numpy(z["grey"][0].astype(np.float32))[None]
            Ic = torch.from_numpy(z["grey"][k].astype(np.float32))[None]
            fl = torch.from_numpy(z["flows"][k].astype(np.float32))[None]
            na, nb, tile, clip = float(z["noise_a"]), float(z["noise_b"]), int(z["tile"]), 0.9
            r0 = analytic_r0(Ir, warp_by_flow(Ic, fl), na, nb, NOMINAL)
            with torch.no_grad():
                feats, _ = compute_features(Ir, Ic, fl, r0, na, nb, clip, tile)
                R, _ = net(feats, r0)
            H, W = Ir.shape[-2:]
            inp, out = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
            with open(inp, "wb") as fh:
                np.array([W, H, tile], np.int32).tofile(fh)
                np.array([na, nb, clip], np.float32).tofile(fh)
                Ir.numpy().astype(np.float32).tofile(fh)
                Ic.numpy().astype(np.float32).tofile(fh)
                fl[0].permute(1, 2, 0).numpy().astype(np.float32).tofile(fh)   # interleaved (dx,dy)
                r0.numpy().astype(np.float32).tofile(fh)
            subprocess.run([exe, inp, out], check=True)
            o = np.fromfile(out, np.float32)
            Rc = o[:H * W].reshape(H, W)
            Fc = o[H * W:].reshape(-1, H, W)
            fd = np.abs(Fc - feats[0].numpy()).reshape(len(NAMES), -1).max(1)
            rd = float(np.abs(Rc - R[0].numpy()).max())
            worst_f, worst_r = max(worst_f, float(fd.max())), max(worst_r, rd)
            print(f"{os.path.basename(f)} k={k} fs={tile}: max|dR|={rd:.2e}  worst feature "
                  f"{NAMES[int(fd.argmax())]} {fd.max():.2e}  R range [{R.min():.3f},{R.max():.3f}]")
    ok = worst_f < 1e-4 and worst_r < 1e-4
    print("PARITY", "OK" if ok else "FAIL", f"(features {worst_f:.2e}, R {worst_r:.2e})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
