"""Where and why a mask net rejects ALIASING on perfectly aligned fine-text bursts (v21 sets).

    RNN_GT_NMERGE=8 python alias_why.py runs/v17/model.pt [runs/v21/model_sXXXX.pt shift]
Perfectly aligned set c21txp_val: every visible pixel's target is "keep"; misaligned twins c21txm_val: same text,
deliberate misalignment. Reports
  WHERE: weight lost on kept-target pixels by aliasing strength (the perfectly aligned frame's residual vs the
         reference, in noise units = aliasing + noise only) and by text vs background, net vs analytic mask;
  WHY:   per-feature means of (A) aliasing pixels the net rejects, (B) misaligned pixels it rejects correctly,
         (C) aliasing pixels it keeps; per-feature separability of A vs B (AUC); and which single feature, set to its
         typical kept value (median of C), lifts the rejected A pixels the most.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                      # noqa: E402
import torch                                            # noqa: E402
import train_pix as tp                                  # noqa: E402
from real_protocol import load_net, stretch             # noqa: E402
from rnn.features import NAMES                          # noqa: E402

LO, HI = 0.2, 0.9


def auc(a, b):
    """P(a > b) for two 1-D samples (ties half)."""
    x = np.concatenate([a, b]); r = x.argsort().argsort().astype(np.float64) + 1
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / max(1, len(a) * len(b))


def collect(net, shift, vs):
    ds = tp.Cache(f"data/{vs}")
    F_, R_, R0_, T_, E_, A_ = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            d = ds[i]
            R = net(d["feats"], d["r0"])[0].clamp(1e-6, 1 - 1e-6)
            R = stretch(torch.sigmoid(torch.log(R) - torch.log1p(-R) + shift), LO, HI)
            F_.append(d["feats"].permute(0, 2, 3, 1).reshape(-1, d["feats"].shape[1]).numpy())
            R_.append(R.reshape(-1).numpy()); R0_.append(d["r0"].reshape(-1).numpy())
            T_.append(d["geo"].reshape(-1).numpy()); E_.append(d["edge"].reshape(-1).numpy())
            A_.append(d["feats"][:, 2].reshape(-1).numpy())          # f2: residual energy vs noise (log)
    return [np.concatenate(x) for x in (F_, R_, R0_, T_, E_, A_)]


def main():
    assert os.environ.get("RNN_GT_NMERGE") == "8"
    torch.set_num_threads(8)
    path = sys.argv[1]; shift = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    net = load_net(os.path.join(HERE, path))
    Fp, Rp, R0p, Tp, Ep, Ap = collect(net, shift, "c21txp_val")
    Fm, Rm, R0m, Tm, Em, Am = collect(net, shift, "c21txm_val")
    good = Tp > 0.8
    print(f"== {path} {shift:+.2f}  perfectly aligned aliased text: {good.sum()} keep-target pixels")
    print(f"   weight lost on keep-target pixels: net {100 * (1 - Rp[good]).mean():.1f}%   "
          f"analytic {100 * (1 - R0p[good]).mean():.1f}%")
    print("   WHERE - by aliasing residual (f2 = log(1 + residual^2 / noise^2), 3x3/5x5 window):")
    for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 99)):
        m = good & (Ap >= lo) & (Ap < hi)
        if m.sum() == 0:
            continue
        print(f"     f2 {lo:>3}-{hi:<3} {100 * m.sum() / good.sum():5.1f}% of pixels | lost: net {100 * (1 - Rp[m]).mean():5.1f}%"
              f"  analytic {100 * (1 - R0p[m]).mean():5.1f}%")
    for name, m in (("on text/edges (edge >= 0.5)", good & (Ep >= 0.5)), ("background (edge < 0.5)", good & (Ep < 0.5))):
        print(f"     {name:28s} {100 * m.sum() / good.sum():5.1f}% | lost: net {100 * (1 - Rp[m]).mean():5.1f}%"
              f"  analytic {100 * (1 - R0p[m]).mean():5.1f}%")
    A = good & (Rp < 0.5); C = good & (Rp >= 0.9); B = (Tm < 0.2) & (Rm < 0.5)
    print(f"   WHY - groups: A aliasing rejected {A.sum()}, B misaligned rejected {B.sum()}, C aliasing kept {C.sum()}")
    print(f"     {'feature':12s} {'A mean':>8s} {'B mean':>8s} {'C mean':>8s} {'AUC A>B':>8s}  lift if set to C median")
    Xa = torch.from_numpy(Fp[A][:20000]); r0a = torch.from_numpy(R0p[A][:20000])
    med = np.median(Fp[C], 0)
    with torch.no_grad():
        def r_of(X):   # the net on single pixels: features at all 9 conv taps = the pixel's own (local constant)
            x = X[:, :, None, None].expand(-1, -1, 3, 3).contiguous()
            R = net(x, r0a[:, None, None].expand(-1, 3, 3))[0][:, 1, 1].clamp(1e-6, 1 - 1e-6)
            return stretch(torch.sigmoid(torch.log(R) - torch.log1p(-R) + shift), LO, HI)
        base = r_of(Xa)
        for c in range(Fp.shape[1]):
            X2 = Xa.clone(); X2[:, c] = float(med[c])
            lift = float((r_of(X2) - base).mean())
            print(f"     {NAMES[c]:12s} {Fp[A][:, c].mean():8.3f} {Fm[B][:, c].mean():8.3f} {Fp[C][:, c].mean():8.3f} "
                  f"{auc(Fp[A][:, c], Fm[B][:, c]):8.2f}  {lift:+.3f}", flush=True)


if __name__ == "__main__":
    main()
