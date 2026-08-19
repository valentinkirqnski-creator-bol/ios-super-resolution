"""Train a robustness mask that predicts merge-safety directly.

Inputs are what any mask could see (reference stats, warped comparison stats,
the estimated flow, its local consistency, expected noise, and the analytic
mask's own answer). The target is the ideal robustness derived from ground
truth: 1 where the alignment fetched the right content, falling off where it
did not. See rob_real.cpp for how a real burst yields that label without
synthesising any imagery, and why it is not circular.

Also scores the classical Wronski mask against the same ground truth, which is
the only apples-to-apples comparison either has had.

Why the loss changed, and why that is the whole point
----------------------------------------------------
The previous version minimised a CLASS-WEIGHTED squared error, upweighting
rejection by `rej_w` because harmful pixels are the minority. That choice, not
the labels, is what produced a mask stuck at mid-grey, and the arithmetic is
worth stating because it was mistaken for a data problem twice.

Weighted MSE is minimised by the WEIGHTED conditional mean. With a fraction f
of harmful pixels at target 0 and (1-f) harmless at target 1, the optimal
prediction for an ambiguous feature vector is

    (1-f) / ((1-f) + f * (1 + rej_w))

At f = 0.05 and rej_w = 8 that is 0.70; at rej_w = 2, 0.86. Neither exceeds
0.9, so the mask CANNOT emit the top of its range no matter how clean the
labels are or how long it trains -- which is exactly what was measured (0.0%
of pixels above 0.9 against the analytic mask's 62-74%). The upweighting was
the cause.

So class weighting is gone by default. Imbalance is handled where it belongs,
in how the generator places crops, and the remaining two changes address the
other half of the problem:

  * BCE rather than MSE. Both are minimised by the conditional mean, so both
    are calibrated in principle, but MSE composed with a sigmoid has a
    gradient carrying sigma'(z) twice over, which vanishes at both ends -- the
    optimiser is pushed hardest in the middle and barely at all where the
    answer is "certainly 1". BCE cancels that factor. This is the ordinary
    reason sigmoid+MSE regressions sit in mid-range.
  * the validity mask is honoured. Pixels the generator could not vouch for
    carry weight 0 and must not contribute a gradient; averaging over them
    would drag every prediction toward whatever those pixels happen to hold.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn

SC = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.environ.get("ROB_DATA") or (
    sys.argv[1] if len(sys.argv) > 1 else os.path.join(SC, "robset"))

IN_CH = 18          # see build_robustness_nn_features in core/stages.h
NCH = 25            # 18 inputs + 7 analysis channels
CH_HARM, CH_IDEAL_R, CH_FERR, CH_W, CH_REP = 18, 19, 20, 21, 22
# The scene-motion label component, kept separate from the corruption harm so
# evaluation can report occlusion and disocclusion on their own. CH_OCC is
# "this frame disagrees with the others here" (background hidden behind a
# moving object in THIS frame); CH_DIS is "every frame disagrees with the
# reference here" (the reference is the odd view -- the moving object was
# there and has since left), which is the one that ghosts a moving subject.
CH_OCC, CH_DIS = 23, 24
CH_ANALYTIC_R = 14  # the classical mask's answer, an INPUT, never a target

PATTERNS = ["none", "single", "neighbours", "group_same", "smooth", "abrupt",
            "rotation", "trans_rot", "magnitude", "direction", "edge_aligned",
            "edge_perp", "similar", "global"]


def load_dataset(prefix):
    """Returns (memmap, meta, records) or (None, {}, []) when absent.

    The dataset is required to TRAIN but not to import RobNet -- export_
    coreml.py only needs the architecture and IN_CH, and demanding a multi-GB
    blob to convert an already-trained checkpoint made the exporter unrunnable.
    """
    meta = {}
    try:
        with open(prefix + ".meta") as f:
            for line in f:
                k, v = line.split()
                meta[k] = int(v)
    except FileNotFoundError:
        return None, {}, []
    gh, gw, nch = meta["guide_h"], meta["guide_w"], meta["channels"]
    assert nch == NCH, f"expected {NCH} channels per record, sidecar says {nch}"
    n = meta["records"]
    data = np.memmap(prefix + ".f32", dtype=np.float32, mode="r",
                     shape=(n, gh, gw, nch))
    recs = []
    with open(prefix + ".idx") as f:
        for line in f:
            p = line.split()
            if len(p) < 12:
                continue
            recs.append(dict(rec=int(p[0]), burst=int(p[1]), ref=int(p[2]),
                             comp=int(p[3]), variant=int(p[4]), pattern=int(p[5]),
                             band=int(p[6]), cy=int(p[7]), cx=int(p[8]),
                             ferr=float(p[9]), bad=float(p[10]), valid=float(p[11]),
                             gain=float(p[12]) if len(p) > 12 else 1.0))
    assert len(recs) == n, f"idx has {len(recs)} rows, meta says {n}"
    return data, meta, recs


def split_records(recs, guide_w=2016, holdout_burst=None):
    """Train/eval split.

    Two independent holdouts, because they answer different questions:

      * REGION. Crops whose origin lies in the right-hand strip of the frame
        are eval-only for every burst. This measures generalisation to unseen
        content within scenes the model has otherwise trained on.
      * BURST. Optionally an entire burst is withheld. This is the question the
        user actually asked -- the previous model did well on the burst it was
        trained on and merged sky into mountain on a new one -- and a
        region holdout cannot answer it, since the two regions share a scene,
        an exposure and a noise level.

    Splitting on the crop ORIGIN rather than on the record index also keeps a
    training crop from overlapping an eval crop, which would leak.
    """
    x_cut = int(0.62 * guide_w)
    train, ev_region, ev_burst = [], [], []
    for r in recs:
        if holdout_burst is not None and r["burst"] == holdout_burst:
            ev_burst.append(r)
        elif r["cx"] > x_cut:
            ev_region.append(r)
        else:
            train.append(r)
    return train, ev_region, ev_burst


# ---------------------------------------------------------------------- model
class RobNet(nn.Module):
    """Depthwise-separable, dilated. Receptive field 7 guide px = ~28 raw px.

    The analytic mask is a pointwise function of a 3x3 statistic, which is why
    it cannot tell a tile that disagrees with its neighbours from one that does
    not. Dilation buys that context without a resolution pyramid.

    Separable rather than dense because this runs per comparison frame over the
    whole guide plane, where cost is set by pixels, not parameters: a dense
    32-channel stack is 26.8k MAC/px -- 490 GMAC for a 6-frame burst. Factoring
    each 3x3 into depthwise + pointwise and halving the width gives ~1.5k
    MAC/px for the same receptive field. Width 16 is also a multiple of the ANE
    tile width, so the channel dimension packs without waste.

    The runtime budget (200 ms for everything the mask adds, feature
    construction included) settles this: the dense variant is not a candidate.
    """
    def __init__(self, cin=IN_CH, w=16):
        super().__init__()
        def block(d):
            return [nn.Conv2d(w, w, 3, padding=d, dilation=d, groups=w),
                    nn.Conv2d(w, w, 1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(
            nn.Conv2d(cin, w, 1), nn.ReLU(inplace=True),
            *block(1), *block(2), *block(4),
            nn.Conv2d(w, 1, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))

    def logits(self, x):
        return self.net(x)


def fit_norm(data, recs, cap=192):
    """Input mean/std over a spread of TRAINING records only.

    Fitted on the training split alone: normalisation constants derived from
    the eval records would leak their statistics into the model, and the flow
    channels in particular differ a lot between bursts.
    """
    step = max(1, len(recs) // cap)
    sub = np.asarray([data[r["rec"], ::4, ::4, :IN_CH] for r in recs[::step]],
                     dtype=np.float32)
    flat = sub.reshape(-1, IN_CH)
    mu = flat.mean(0)
    sd = flat.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def classical_R(px):
    """The analytic mask, read straight from channel 14.

    This used to re-derive Eq. 5-8 in numpy, which made it a THIRD
    implementation alongside the C++ port and the generator -- and it quietly
    used its own s1/s2/t/Mt rather than the ones the burst was processed with,
    so the baseline it reported was not the mask the user runs. The generator
    now writes the real value via the shared robustness_analytic_R, so the
    honest baseline is simply to read it.
    """
    return px[..., CH_ANALYTIC_R]


def main():
    data, meta, recs = load_dataset(PREFIX)
    if data is None:
        raise FileNotFoundError(f"no dataset at {PREFIX}")
    gh, gw = meta["guide_h"], meta["guide_w"]
    holdout_burst = os.environ.get("ROB_HOLDOUT_BURST")
    holdout_burst = int(holdout_burst) if holdout_burst else None
    train, ev_region, ev_burst = split_records(recs, holdout_burst=holdout_burst)
    print(f"dataset: {len(recs)} records of {gh}x{gw}x{meta['channels']}")
    print(f"  train {len(train)}   eval-region {len(ev_region)}   "
          f"eval-burst {len(ev_burst)}"
          + (f" (burst {holdout_burst} withheld entirely)" if holdout_burst else ""))
    if not train:
        raise SystemExit("empty training split")

    mu, sd = fit_norm(data, train)
    print("input mean:", np.array2string(mu, precision=3))
    print("input std :", np.array2string(sd, precision=3))

    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    model = RobNet()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model: {nparam} parameters")

    ps = int(os.environ.get("ROB_PATCH", min(96, gh, gw)))
    bs = int(os.environ.get("ROB_BATCH", 24))
    steps = int(os.environ.get("ROB_STEPS", 3000))
    # Class weighting defaults OFF. See the module docstring: it is what pinned
    # the previous mask below 0.9 everywhere. Exposed only so the trade can be
    # re-measured, never as a default.
    rej_w = float(os.environ.get("ROB_REJ_W", 0.0))

    def sample_batch():
        xs, ys, ws = [], [], []
        for _ in range(bs):
            r = train[rng.randint(len(train))]
            y0 = rng.randint(max(1, gh - ps))
            x0 = rng.randint(max(1, gw - ps))
            p = np.asarray(data[r["rec"], y0:y0 + ps, x0:x0 + ps, :], dtype=np.float32)
            xs.append((p[..., :IN_CH] - mu) / sd)
            ys.append(p[..., CH_IDEAL_R:CH_IDEAL_R + 1])
            ws.append(p[..., CH_W:CH_W + 1])
        t = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2)
        return t(xs), t(ys), t(ws)

    bce = nn.BCEWithLogitsLoss(reduction="none")

    # The user's cost is ASYMMETRIC and the two halves of it pull in opposite
    # directions, so the loss has to carry both at once.
    #
    #   a merged misalignment is unacceptable  -> false ACCEPTANCE costs FA_W x
    #   a forgone merge is merely wasteful     -> false rejection costs 1x
    #
    # but also, and this is the part a plain reweighting does NOT give:
    #
    #   a provably-safe pixel must reach ~1.0, not 0.98. Measured on the
    #   symmetric model, zero-harm pixels averaged 0.9795 and only 64.3% cleared
    #   a 0.989 gate, so that gate threw away 35.7% of pixels that were
    #   perfectly safe to merge. Rescaling cannot fix that: temperature and
    #   isotonic are monotone, so they preserve the ranking and merely slide
    #   along the same trade-off curve (T=0.5 at gate 0.989 lands on the same
    #   operating point as gate 0.90 with no temperature). The curve has to be
    #   LIFTED, which means margin, not scale.
    #
    # Hence the two hinge terms below. They act on the LOGIT and have constant
    # gradient until their margin is met, so unlike BCE -- whose gradient is
    # (p - target) and so fades to 0.02 exactly where the model is stalling --
    # they keep pushing at the top and bottom of the range.
    FA_W = float(os.environ.get("ROB_FA_W", 6.0))     # false-acceptance penalty
    SAFE_W = float(os.environ.get("ROB_SAFE_W", 3.0))  # confidence on safe pixels
    M_POS = float(os.environ.get("ROB_M_POS", 6.0))   # logit 6 = 0.9975
    M_NEG = float(os.environ.get("ROB_M_NEG", -2.0))  # logit -2 = 0.12

    def loss_fn(logits, target, w):
        # Validity first: pixels the generator could not vouch for are unknown,
        # not "probably fine", and must contribute no gradient at all.
        # Harm MAGNITUDE, recovered from the label. r_ideal = exp(-z^2/4) with z
        # the mis-fetch in units of sensor sigma, so z = 2*sqrt(-ln r). Weighting
        # by it makes visibly-wrong pixels dominate the objective and stops
        # near-zero-harm cases -- which are numerous and invisible -- from
        # consuming capacity.
        z = 2.0 * torch.sqrt(torch.clamp(-torch.log(target.clamp(1e-8, 1.0)), min=0.0))
        mag = (z / 2.0).clamp(0.0, 4.0)
        ww = w * (1.0 + FA_W * mag)
        l = (bce(logits, target) * ww).sum() / ww.sum().clamp_min(1.0)

        # Margin: provably-safe pixels must clear the gate with headroom.
        safe = (target > 0.999).float() * w
        if float(safe.sum()) > 0:
            l = l + SAFE_W * ((M_POS - logits).clamp(min=0.0) * safe).sum()                     / safe.sum().clamp_min(1.0)
        # Margin: visibly harmful pixels must sit well below it. Without this,
        # pushing the safe end up simply drags everything up and false
        # acceptance rises at a fixed gate -- the failure mode to watch for.
        bad = (target < 0.1).float() * w
        if float(bad.sum()) > 0:
            l = l + FA_W * ((logits - M_NEG).clamp(min=0.0) * bad).sum()                     / bad.sum().clamp_min(1.0)
        return l

    lr0 = float(os.environ.get("ROB_LR", 3e-3))
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    for it in range(steps):
        for g in opt.param_groups:
            # Cosine decay to zero. What is being fitted is a VALUE the merge
            # multiplies by, not a ranking; holding lr flat to the last step
            # leaves the weights bouncing around the minimum, which shows up
            # as a squashed output range.
            g["lr"] = lr0 * 0.5 * (1.0 + np.cos(np.pi * it / max(1, steps - 1)))
        x, y, w = sample_batch()
        loss = loss_fn(model.logits(x), y, w)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, steps // 15) == 0 or it == steps - 1:
            print(f"  step {it:5d}  loss {loss.item():.5f}")

    ckpt_path = os.environ.get("ROB_OUT", os.path.join(SC, "robnet.pt"))
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd,
                "in_ch": IN_CH, "holdout_burst": holdout_burst}, ckpt_path)
    with open(os.path.splitext(ckpt_path)[0] + "_norm.json", "w") as f:
        json.dump({"mu": mu.tolist(), "sd": sd.tolist(), "in_ch": IN_CH}, f)
    print("\nsaved", ckpt_path)
    print("run eval_rob.py for the scored comparison against the analytic mask")


if __name__ == "__main__":
    main()
