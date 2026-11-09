"""Random-weights parity: (1) write a random CrefNet + its header (mode 'make'), (2) run PyTorch on the
dumped full-frame features and compare with the CPU / Metal-shim outputs (mode 'check').
usage: python cref_parity.py make outdir   |   python cref_parity.py check outdir H W"""
import os
import subprocess
import sys
import numpy as np
import torch
from train_cref import CrefNet
from crops import NF

mode, d = sys.argv[1], sys.argv[2]
if mode == "make":
    torch.manual_seed(1)
    m = CrefNet()
    for c in m.convs:
        torch.nn.init.normal_(c.weight, 0, 0.3); torch.nn.init.normal_(c.bias, 0, 0.1)
    torch.nn.init.normal_(m.head.weight, 0, 0.5); torch.nn.init.normal_(m.head.bias, 0, 0.2)
    os.makedirs(os.path.join(d, "inc"), exist_ok=True)
    torch.save(m.state_dict(), os.path.join(d, "rand.pt"))
    subprocess.check_call([sys.executable, "-B", "export_weights.py", os.path.join(d, "rand.pt"),
                           os.path.join(d, "inc", "cref_weights.h")])
else:
    H, W = int(sys.argv[3]), int(sys.argv[4])
    m = CrefNet()
    m.load_state_dict(torch.load(os.path.join(d, "rand.pt")))
    F = np.fromfile(os.path.join(d, "feats.f32"), np.float32).reshape(1, NF, H, W)
    with torch.no_grad():
        R = m(torch.from_numpy(F))[0, 0].numpy()
    rc = np.fromfile(os.path.join(d, "rcpu.f32"), np.float32).reshape(H, W)
    rm = np.fromfile(os.path.join(d, "rmetal.f32"), np.float32).reshape(H, W)
    s = slice(8, -8)   # top/bottom 8 rows: the band halo is edge-clamped features, not per-layer replicate
    print("torch vs CPU   interior max |diff| %.3g  (all rows %.3g)" % (np.abs(R - rc)[s].max(), np.abs(R - rc).max()))
    print("torch vs Metal interior max |diff| %.3g" % np.abs(R - rm)[s].max())
    print("R' range %.3f..%.3f  mean change vs input R %.3f" % (R.min(), R.max(), np.abs(R - F[0, 0]).mean()))
