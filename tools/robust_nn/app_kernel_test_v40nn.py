"""nn-mask robust_nn v40 Metal kernels on the CPU (cpp/app_kernel_emu_v40nn.cpp compiles the exact block appended to
<core>/HHSRKernels.metal) vs PyTorch, on cached app-stage samples (non-homography sets).

    python app_kernel_test_v40nn.py --core <nn-mask>/core [--model runs/v40b/model.pt]
Default: a random 21-input net. The comparison grey is fed as RAW quads with random un-prewhiten factors, so the
raw-to-grey kernel is tested too. Prints KERNELS OK / FAIL."""
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
from rnn.features import compute_features, motion_structure_features   # noqa: E402
from rnn.model import RobustNet                                         # noqa: E402
from real_protocol import load_net, stretch, CLIP                       # noqa: E402
from export_weights import write_header                                 # noqa: E402

MARK = "// robust_nn v40 (Config::robust_nn_enabled): learned refinement of the"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--sets", default="c40M_val,c40A_val,c15_val,c21txm_val")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.9)
    a = ap.parse_args()
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    if a.model:
        net = load_net(a.model)
    else:
        net = RobustNet(nf=21)
        with torch.no_grad():
            for p_ in net.parameters():
                p_.normal_(0, 0.3)
            net.mu.normal_(0, 0.5)
            net.sd.uniform_(0.5, 1.5)
    net.eval()
    mot = net.c1.weight.shape[1] - 14
    src = open(os.path.join(a.core, "HHSRKernels.metal"), encoding="utf-8").read()
    block = src[src.index(MARK):]
    worst_g, worst_r = 0.0, 0.0
    with tempfile.TemporaryDirectory() as d:
        hdr = os.path.join(d, "w.h").replace("\\", "/")
        write_header(net, hdr, "kernel test v40 nn-mask")
        blk = os.path.join(d, "block.metal").replace("\\", "/")
        open(blk, "w", encoding="utf-8").write(block)
        exe = os.path.join(d, "emu.exe")
        subprocess.run(["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-static", "-I", a.core,
                        f'-DRNN_WEIGHTS_HEADER="{hdr}"', f'-DRNN_APP_BLOCK="{blk}"',
                        os.path.join(HERE, "cpp", "app_kernel_emu_v40nn.cpp"), "-o", exe], check=True)
        for sname in a.sets.split(","):
            for fn in sorted(glob.glob(os.path.join(HERE, "data", sname, "*.npz")))[:a.n]:
                z = np.load(fn)
                Ir, G = torch.from_numpy(z["ref"]), torch.from_numpy(z["grey"])
                fl, r0 = torch.from_numpy(z["flow"]), torch.from_numpy(z["r0"].astype(np.float32))
                K, (H, W) = G.shape[0], G.shape[1:]
                aa, bb, fs = float(z["a"]), float(z["b"]), max(1, int(z["tile"]) // 2)
                ny, nx = -(-H // fs), -(-W // fs)
                # one vector per flow tile (raw px), and the dense guide-px field the net would see from it
                tfl = np.zeros((K, ny, nx, 2), np.float32)
                for ty in range(ny):
                    for tx in range(nx):
                        tfl[:, ty, tx] = 2.0 * fl[:, :, min(ty * fs, H - 1), min(tx * fs, W - 1)].numpy()
                dense = torch.from_numpy(np.ascontiguousarray(
                    np.repeat(np.repeat(tfl, fs, 1), fs, 2)[:, :H, :W] / 2.0)).permute(0, 3, 1, 2).contiguous()
                Ik = Ir[None].expand(K, -1, -1).contiguous()
                with torch.no_grad():
                    f14, J = compute_features(Ik, G, dense, r0, aa, bb, CLIP, fs)
                    feats = torch.cat([f14[:, :14], motion_structure_features(Ik, J, dense, r0, aa, bb, fs)[:, :mot]], 1)
                    R, _ = net(feats, r0)
                    Rs = stretch(R, a.lo, a.hi)
                for k in range(K):
                    u = rng.uniform(0.4, 2.5, 4).astype(np.float32)
                    e = rng.normal(0, 0.05, (H, W, 4)).astype(np.float32)
                    e -= e.mean(-1, keepdims=True)
                    g = G[k].numpy()[..., None] * (1.0 + e) / u          # quad mean of raw * u == grey
                    raw = np.zeros((2 * H, 2 * W), np.float32)
                    raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2] = (g[..., i] for i in range(4))
                    pi, po = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
                    with open(pi, "wb") as fh:
                        np.array([W, H, fs], np.int32).tofile(fh)
                        np.array([aa, bb, CLIP, a.lo, a.hi, *u], np.float32).tofile(fh)
                        for t in (raw, Ir.numpy(), tfl[k], r0[k].numpy()):
                            np.ascontiguousarray(t, np.float32).tofile(fh)
                    subprocess.run([exe, pi, po], check=True)
                    out = np.fromfile(po, np.float32).reshape(2, H, W)
                    worst_g = max(worst_g, float(np.abs(out[0] - G[k].numpy()).max()))
                    worst_r = max(worst_r, float(np.abs(out[1] - Rs[k].numpy()).max()))
                print(f"{sname}/{os.path.basename(fn)}: grey {worst_g:.2e}  R {worst_r:.2e}", flush=True)
    ok = worst_g < 1e-5 and worst_r < 2e-3
    print("KERNELS OK" if ok else "KERNELS FAIL", f"(grey {worst_g:.2e}, R {worst_r:.2e})")


if __name__ == "__main__":
    main()
