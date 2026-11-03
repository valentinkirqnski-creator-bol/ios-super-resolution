"""v40 device parity: core/robust_nn_shared.h (14 features + the RNN_MOT per-frame motion-prior / structure inputs,
and the net) compiled as C++ vs rnn/features.py + RobustNet on real cached app-stage samples.

    python parity_v40.py --core <repo>/core [--model runs/v40b/model.pt] [--sets c40M_val,c40A_val,c15_val,c21txm_val]
Default model: a random 21-input net (every weight exercised). Prints PARITY OK / FAIL."""
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
from real_protocol import load_net, CLIP                                # noqa: E402
from export_weights import write_header                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--sets", default="c40M_val,c40A_val,c15_val,c21txm_val")
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    torch.manual_seed(3)
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
    nf = net.c1.weight.shape[1]
    mot = nf - 14
    worst_f, worst_r, flips, tot = np.zeros(nf), 0.0, 0, 0
    with tempfile.TemporaryDirectory() as d:
        hdr = os.path.join(d, "w.h").replace("\\", "/")
        write_header(net, hdr, "parity_v40")
        exe = os.path.join(d, "fd.exe")
        subprocess.run(["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-static", f'-DRNN_WEIGHTS_HEADER="{hdr}"',
                        "-I", a.core, os.path.join(HERE, "cpp", "feat_dump_v40.cpp"), "-o", exe], check=True)
        for sname in a.sets.split(","):
            for fn in sorted(glob.glob(os.path.join(HERE, "data", sname, "*.npz")))[:a.n]:
                z = np.load(fn)
                Ir, G = torch.from_numpy(z["ref"]), torch.from_numpy(z["grey"])
                fl, r0 = torch.from_numpy(z["flow"]), torch.from_numpy(z["r0"].astype(np.float32))
                K, (H, W) = G.shape[0], G.shape[1:]
                aa, bb, fs = float(z["a"]), float(z["b"]), max(1, int(z["tile"]) // 2)
                Ik = Ir[None].expand(K, -1, -1).contiguous()
                with torch.no_grad():
                    f14, J = compute_features(Ik, G, fl, r0, aa, bb, CLIP, fs)
                    feats = torch.cat([f14[:, :14], motion_structure_features(Ik, J, fl, r0, aa, bb, fs)[:, :mot]], 1)
                    R, _ = net(feats, r0)
                for k in range(K):
                    pi, po = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
                    with open(pi, "wb") as fh:
                        np.array([W, H, fs], np.int32).tofile(fh)
                        np.array([aa, bb, CLIP], np.float32).tofile(fh)
                        for t in (Ir, J[k], fl[k].permute(1, 2, 0), r0[k]):
                            np.ascontiguousarray(t.numpy(), np.float32).tofile(fh)
                    subprocess.run([exe, pi, po], check=True)
                    out = np.fromfile(po, np.float32)
                    fc = out[:nf * H * W].reshape(nf, H, W)
                    rc = out[nf * H * W:].reshape(H, W)
                    fp = feats[k].numpy()
                    diff = np.abs(fc - fp)
                    if mot >= 2:   # the hard M > 0.8 flag: count flips separately (ties at the threshold)
                        fl15 = diff[15] > 0.5
                        flips += int(fl15.sum()); tot += fl15.size
                        for c in (15, 17):
                            diff[c][fl15] = 0.0
                    worst_f = np.maximum(worst_f, diff.reshape(nf, -1).max(1))
                    worst_r = max(worst_r, float(np.abs(rc - R[k].numpy()).max()))
                print(f"{sname}/{os.path.basename(fn)}: worst feature diff so far {worst_f.max():.2e} "
                      f"(channel {int(worst_f.argmax())}), worst R diff {worst_r:.2e}, flag flips {flips}", flush=True)
    print("per-channel worst:", " ".join(f"{v:.1e}" for v in worst_f))
    ok = worst_f.max() < 2e-3 and worst_r < 2e-3 and flips <= 1e-5 * max(tot, 1)
    print("PARITY OK" if ok else "PARITY FAIL", f"(feature {worst_f.max():.2e}, R {worst_r:.2e}, flips {flips}/{tot})")


if __name__ == "__main__":
    main()
