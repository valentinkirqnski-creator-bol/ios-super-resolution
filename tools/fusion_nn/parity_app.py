"""The app's CPU network (core/fusion_nn.cpp via build/fnn_parity) against PyTorch.

    python parity_app.py <model.pt> <burst.npz> [--time]

Builds the GAT guides exactly as fusion.net_inputs does, runs both, compares logits for the
reference and every comparison frame, and the resulting weights relative to the reference.
--time: also times the C++ network on a 12 MP frame (1512x2016 guide) with 1 and 6 threads.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

import fusion as fu

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "fnn_parity.exe")


def run_cpp(ref, frm, mean, is_ref, a, b, threads=6):
    gh, gw = ref.shape[:2]
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.bin"), os.path.join(td, "out.bin")
        with open(fi, "wb") as f:
            np.array([gh, gw, int(is_ref), threads], np.int32).tofile(f)
            np.array([a, b], np.float32).tofile(f)
            for x in (ref, frm, mean):
                np.ascontiguousarray(x, np.float32).tofile(f)
        r = subprocess.run([EXE, fi, fo], capture_output=True, text=True, check=True)
        return np.fromfile(fo, np.float32).reshape(gh, gw), float(r.stderr.strip())


def main():
    model, path = sys.argv[1], sys.argv[2]
    net = fu.FusionNet(c=12, cin=8)
    net.load_state_dict(torch.load(model))
    net.eval()
    bu = fu.Burst(path)
    gd = fu.warp_guides(fu.guides(bu.frames, bu.cfa), bu)
    g = fu.gat(gd.clamp(min=0), bu.a, bu.b)                       # [N, 3, gh, gw]
    mean = g.mean(0)
    with torch.no_grad():
        lt = net(fu.net_inputs(bu, 8))[:, 0].numpy()             # [N, gh, gw]
    hwc = lambda t: t.permute(1, 2, 0).numpy()
    lc = []
    for f in range(bu.N):
        l, _ = run_cpp(hwc(g[0]), hwc(g[f]), hwc(mean), f == 0, bu.a, bu.b)
        lc.append(l)
    lc = np.stack(lc)
    d = np.abs(lc - lt)
    rt = np.exp(lt[1:] - lt[:1])
    rc = np.exp(lc[1:] - lc[:1])
    print("logits: max |diff| %.2e  mean %.2e   (torch range %.2f..%.2f)" % (d.max(), d.mean(), lt.min(), lt.max()))
    print("R = exp(l_f - l_ref): max rel diff %.2e;  R > 1 on %.1f%% of guide px, max R %.2f" % (
        (np.abs(rc - rt) / rt).max(), 100 * (rt > 1).mean(), rt.max()))
    if "--time" in sys.argv:
        rng = np.random.default_rng(0)
        big = [rng.uniform(0, 100, (1512, 2016, 3)).astype(np.float32) for _ in range(3)]
        for th in (1, 6):
            _, s = run_cpp(*big, False, bu.a, bu.b, th)
            print("12 MP frame (1512x2016 guide), %d thread(s): %.2f s" % (th, s))


if __name__ == "__main__":
    main()
