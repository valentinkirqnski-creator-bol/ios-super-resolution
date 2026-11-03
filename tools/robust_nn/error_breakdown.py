"""Where are the remaining mistakes vs the ground-truth mask? Splits held-out errors into
   over-rejection (truth: aligned, t > 0.8, mask R < 0.5)  - halo next to a real rejection vs false alarm in a clean area
   misses         (truth: misaligned, t < 0.2, mask R > 0.5) - by true flow error (raw px) and edge strength
Masks scored as shipped (exact 0/1 stretch). Uses the same val samples/seeds as training's evaluation.

    python error_breakdown.py runs/v6/model.pt runs/v12/model.pt runs/v13/r02/model.pt
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["RNN_GEO"] = "1"
os.environ.setdefault("RNN_TP_SERVER", "1")
os.environ.setdefault("RNN_TP_REAL", os.path.join(HERE, "build", "pipe", "tp_real_new.exe"))
sys.path.insert(0, HERE)
import glob                                   # noqa: E402
import numpy as np                            # noqa: E402
import torch                                  # noqa: E402
import torch.nn.functional as F               # noqa: E402
from real_protocol import run_tp_real, geo_target, compute_features, load_net, stretch, CLIP   # noqa: E402

FERR_BINS = [(0, 0.3, "<0.3 px"), (0.3, 1, "0.3-1 px"), (1, 3, "1-3 px"), (3, 1e9, ">3 px")]
EDGE_BINS = [(0, 3, "flat"), (3, 10, "texture"), (10, 1e9, "strong edge")]


def main():
    torch.set_num_threads(2)
    nets = {p.replace("\\", "/").split("runs/")[1].split("/model")[0].replace("/", " "): load_net(p)
            for p in sys.argv[1:]}
    files = sorted(glob.glob(os.path.join(os.environ.get("VAL", "data/rp13_val"), "*.npz")))
    n = len(files)
    S = {}
    tot = {"good": 0, "bad": 0}

    def add(k, key, v):
        S.setdefault(k, {}).setdefault(key, 0.0)
        S[k][key] += float(v)

    for idx in range(2 * n):
        z = np.load(files[idx % n])
        raw, gt = z["raw"], z["gt_flow"]
        N = raw.shape[0]
        rng = np.random.default_rng(99 + idx)                  # same seeds as RP(val)
        seed, corrupt = int(rng.integers(1, 1 << 30)), idx >= n
        d = run_tp_real(raw, gt, float(z["alpha"]), float(z["beta"]), seed, corrupt, z["cfa"], z["wb"], nomerge=True)
        a_g, b_g = float(z["alpha"]) / 4.0, float(z["beta"]) / 4.0
        Ir = d["ref_grey"][None].expand(N - 1, -1, -1).contiguous()
        with torch.no_grad():
            feats, _ = compute_features(Ir, d["grey"], d["flow"], d["r0"], a_g, b_g, CLIP, max(1, d["tile"] // 2))
        clean = torch.from_numpy(z["clean_grey"])
        t = geo_target(clean, d["flow"], torch.from_numpy(gt[1:]), torch.from_numpy(z["vis"][1:].astype(np.float32)),
                       float(z["alpha"]), float(z["beta"]))
        ferr = 2.0 * (d["flow"] - torch.from_numpy(gt[1:])).norm(dim=1)          # raw px
        # edge strength of the true content under the comparison sample, in noise sigmas (3x3 max of |grad|)
        c0 = clean[0]
        gx = torch.zeros_like(c0); gy = torch.zeros_like(c0)
        gx[:, 1:-1] = 0.5 * (c0[:, 2:] - c0[:, :-2]); gy[1:-1] = 0.5 * (c0[2:] - c0[:-2])
        sig = ((float(z["alpha"]) * c0.clamp_min(0) + float(z["beta"])) / 4.0).sqrt() + 1e-6
        edge = F.max_pool2d(((gx * gx + gy * gy).sqrt() / sig)[None, None], 3, 1, 1)[0, 0].expand_as(t)
        bad, good = t < 0.2, t > 0.8
        near_bad = F.max_pool2d(bad.float()[:, None], 5, 1, 2)[:, 0] > 0           # within 2 guide px of a misalignment
        cm = d["corr"] > 0.5
        tot["good"] += int(good.sum()); tot["bad"] += int(bad.sum())
        masks = {"analytic": d["r0"]}
        with torch.no_grad():
            for k, net in nets.items():
                masks[k] = stretch(net(feats[:, :net.c1.in_channels], d["r0"])[0])
        for k, R in masks.items():
            over, miss = good & (R < 0.5), bad & (R > 0.5)
            add(k, "over", over.sum()); add(k, "miss", miss.sum())
            add(k, "over halo", (over & near_bad).sum()); add(k, "over clean", (over & ~near_bad).sum())
            add(k, "leak", R[bad].sum()); add(k, "lost", (1 - R[good]).sum())
            add(k, "absdiff", (R - t).abs().sum()); add(k, "npix", t.numel())
            for lo, hi, nm in FERR_BINS:
                b = bad & (ferr >= lo) & (ferr < hi)
                add(k, "bad " + nm, b.sum()); add(k, "miss " + nm, (miss & b).sum())
            for lo, hi, nm in EDGE_BINS:
                b = bad & (edge >= lo) & (edge < hi)
                add(k, "bad " + nm, b.sum()); add(k, "miss " + nm, (miss & b).sum())
                g = good & (edge >= lo) & (edge < hi)
                add(k, "good " + nm, g.sum()); add(k, "over " + nm, (over & g).sum())
            sm = bad & (ferr < 1) & (edge >= 10)                                    # small shift on a strong edge
            add(k, "bad small-edge", sm.sum()); add(k, "miss small-edge", (miss & sm).sum())
            add(k, "bad corrupted", (bad & cm).sum()); add(k, "miss corrupted", (miss & cm).sum())
            add(k, "bad natural", (bad & ~cm).sum()); add(k, "miss natural", (miss & ~cm).sum())
        print(f"{idx + 1}/{2 * n}", flush=True)

    G, B = tot["good"], tot["bad"]
    print(f"\nheld-out pixels: aligned {G}, misaligned {B}")
    for k, s in S.items():
        err = s["over"] + s["miss"]
        pct = lambda a, b: 100.0 * s[a] / max(1.0, s[b])
        bal = 50 * ((1 - s["miss"] / B) + (1 - s["over"] / G))
        print(f"\n== {k}: accuracy {100 * (1 - err / (G + B)):.2f}%  balanced {bal:.2f}%  similarity "
              f"{1 - s['absdiff'] / s['npix']:.4f}  leak {s['leak'] / B:.3f}  keep {1 - s['lost'] / G:.3f}")
        print(f"   errors: over-rejection {100 * s['over'] / err:.0f}% of mistakes, misses {100 * s['miss'] / err:.0f}%")
        print(f"   over-rejected aligned pixels {100 * s['over'] / G:.2f}%  (halo next to a real misalignment "
              f"{100 * s['over halo'] / G:.2f}%, false alarm in clean area {100 * s['over clean'] / G:.2f}%)")
        print("   over-rejection by content: " + ", ".join(f"{nm} {pct('over ' + nm, 'good ' + nm):.1f}%"
                                                         for _, _, nm in EDGE_BINS))
        print(f"   missed misaligned pixels {100 * s['miss'] / B:.2f}%")
        print("   miss rate by true shift:  " + ", ".join(f"{nm} {pct('miss ' + nm, 'bad ' + nm):.1f}% "
                                                         f"(n={int(s['bad ' + nm])})" for _, _, nm in FERR_BINS))
        print("   miss rate by content:     " + ", ".join(f"{nm} {pct('miss ' + nm, 'bad ' + nm):.1f}%"
                                                         for _, _, nm in EDGE_BINS))
        print(f"   small shift (<1 px) on strong edge: miss {pct('miss small-edge', 'bad small-edge'):.1f}% "
              f"(n={int(s['bad small-edge'])});  corrupted tiles miss {pct('miss corrupted', 'bad corrupted'):.1f}%, "
              f"natural misalignments miss {pct('miss natural', 'bad natural'):.1f}%")


if __name__ == "__main__":
    main()
