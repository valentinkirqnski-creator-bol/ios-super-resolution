"""Is a net BETTER, or just set to a different keep/reject balance? Shifts each net's output logit by a bias b
(exactly what adding b to the last layer's bias does - ships as a weights-only change) and prints, per bias,
missed / over-rejected / small-shift missed / rotation missed / flat over-rejected on the held-out set.
Compare nets at the SAME miss rate.

    python curve_check.py runs/v13/r03/model.pt runs/v14/model_s200.pt
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for k, v in dict(RNN_GEO="1", RNN_NOMERGE="1", RNN_TP_SERVER="1", RNN_OCC_LOOK="1", RNN_HARD="1").items():
    os.environ.setdefault(k, v)
sys.path.insert(0, HERE)
import numpy as np                      # noqa: E402
import torch                            # noqa: E402
from real_protocol import RP, load_net, stretch   # noqa: E402

BIASES = [float(x) for x in os.environ.get("BIASES", "-1.25,-1.0,-0.875,-0.75,-0.625,-0.5,-0.375,-0.25,0").split(",")]
STRETCH = [tuple(float(y) for y in x.split(":")) for x in os.environ.get("STRETCH", "0.05:0.95").split(",")]


def main():
    torch.set_num_threads(2)
    nets = {p.replace("\\", "/").split("runs/")[1].replace("/model", " ").replace(".pt", ""): load_net(p)
            for p in sys.argv[1:]}
    ds = RP(os.path.join(HERE, "data", "rp13_val"), False)
    C, G = {}, np.zeros(2)
    for i in range(len(ds)):
        d = ds[i]
        t = d["geo"]
        bad, good = t < 0.2, t > 0.8
        se, ro, fl = bad & (d["ferr"] < 1), bad & d["rot"], good & (d["edge"][None] < 0.5)
        G = G + np.array([float(stretch(t)[bad].sum()), float(bad.sum())])
        with torch.no_grad():
            for k, net in nets.items():
                R = net(d["feats"][:, :net.c1.in_channels], d["r0"])[0].clamp(1e-6, 1 - 1e-6)
                lg = torch.log(R) - torch.log1p(-R)
                for b in BIASES:
                    for lo, hi in STRETCH:
                        Rs = stretch(torch.sigmoid(lg + b), lo, hi)
                        miss, over = bad & (Rs >= 0.5), good & (Rs <= 0.5)
                        v = np.array([float(x.sum()) for x in (miss, bad, over, good, miss & se, se, miss & ro, ro,
                                                               over & fl, fl, Rs[bad].sum(), Rs[t < 0.05].sum(),
                                                               (t < 0.05).sum(), (1 - Rs[good]).sum())])
                        C[(k, b, lo, hi)] = C.get((k, b, lo, hi), 0) + v
        print(f"{i + 1}/{len(ds)}", flush=True)
    print(f"\n{'net':14s} {'bias':>5s} {'lo':>4s} {'hi':>4s} {'balanced':>8s} {'missed':>7s} {'over':>6s} {'small-sh':>8s} "
          f"{'rotation':>8s} {'flat-over':>9s} {'leak':>6s} {'leak<.05':>8s} {'keep':>6s}")
    for (k, b, lo, hi), v in C.items():
        q = lambda j: 100 * v[j] / max(1, v[j + 1])
        print(f"{k:14s} {b:5.2f} {lo:4.2f} {hi:4.2f} {100 - 0.5 * (q(0) + q(2)):8.2f} {q(0):7.2f} {q(2):6.2f} "
              f"{q(4):8.2f} {q(6):8.2f} {q(8):9.2f} {v[10] / v[1]:6.3f} {v[11] / max(1, v[12]):8.3f} "
              f"{1 - v[13] / v[3]:6.3f}")
    print(f"GT mask's own leak (weight it gives its misaligned pixels): {G[0] / G[1]:.3f}")


if __name__ == "__main__":
    main()
