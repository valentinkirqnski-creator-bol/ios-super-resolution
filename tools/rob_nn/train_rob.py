"""Train the robustness CORRECTION: a confidence C in [0,1] the analytic mask
is multiplied by.

    R_final = R_normal * C_nn

The network does not emit the mask and cannot raise it. C comes out of a
sigmoid and the merge multiplies, so "the correction can only ever lower R" is
a property of the graph rather than something this loss is asked to deliver.
That matters because the analytic mask is right most of the time, and the
previous design -- network REPLACES Eq. 5-9 -- made every weakness of the model
a regression in regions the closed form already handled.

The target, and why it is a ratio
---------------------------------
Supervising C against "is this pixel safe" would waste the whole model on
pixels the closed form already rejects. What is wanted is the residual: where
is R_normal too HIGH. So with R_ideal the harm-derived ideal robustness (see
rob_real.cpp) and R_normal the analytic mask's own finished answer,

    C_target = clip((R_ideal + eps) / (R_normal + eps), 0, 1)

  * R_normal already at or below R_ideal  ->  C_target = 1, nothing to do.
  * R_normal = 1 on a pixel merging would damage  ->  C_target ~ R_ideal.
  * R_normal already 0  ->  the ratio is undefined in principle and irrelevant
    in practice, which the eps and the impact weight below handle together.

and the loss is weighted by R_normal, because the error this model makes in
the final mask is R_normal * (C - C_target). A pixel the analytic mask has
already thrown away cannot be made worse or better by C, and should not
consume capacity.

The failure mode to design against
----------------------------------
A model that lowers C everywhere would score well on detection and be
worthless: it has rescaled the mask, not corrected it. Three things guard
against that, and the third is the one that actually decides it:

  1. the ratio target is exactly 1.0 wherever the analytic mask is not too
     high, which is the overwhelming majority of pixels;
  2. the output bias starts at +5 (C = 0.993), so "do nothing" is the
     initialisation and departures from it have to be earned;
  3. a hinge that keeps the logit above M_POS on provably-safe pixels, with
     constant gradient. BCE's gradient is (p - target) and fades to nothing
     exactly where the model is drifting from 0.999 to 0.98 -- a drift that is
     invisible in the loss and very visible in the mask.

Reported at the end: the distribution of C on correctly-aligned pixels. If its
mean is not within a hair of 1, the model is a failure regardless of what its
detection numbers say.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn

SC = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.environ.get("ROB_DATA") or (
    sys.argv[1] if len(sys.argv) > 1 else os.path.join(SC, "robset"))

IN_CH = 27          # see build_robustness_nn_features in core/stages.h
NCH = 35            # 27 inputs + 8 analysis channels
CH_HARM, CH_IDEAL_R, CH_FERR, CH_W, CH_REP = 27, 28, 29, 30, 31
# The scene-motion label component, kept separate from the corruption harm so
# evaluation can report occlusion and disocclusion on their own. CH_OCC is
# "this frame disagrees with the others here" (background hidden behind a
# moving object in THIS frame); CH_DIS is "every frame disagrees with the
# reference here" (the reference is the odd view -- the moving object was
# there and has since left), which is the one that ghosts a moving subject.
CH_OCC, CH_DIS = 32, 33
# The analytic mask's FINISHED answer, after Eq. 9's 5x5 minimum: the value the
# app multiplies. Not to be confused with input channel 14, which is the
# pointwise Eq. 5 before that minimum and is evidence, not the divisor.
CH_RNORM = 34
CH_ANALYTIC_R = 14

PATTERNS = ["none", "single", "neighbours", "group_same", "smooth", "abrupt",
            "rotation", "trans_rot", "magnitude", "direction", "edge_aligned",
            "edge_perp", "similar", "global"]

# Below this the analytic mask has already all but rejected the pixel, so the
# ratio target is numerically unstable and practically irrelevant.
TGT_EPS = 0.05


def target_and_weight(px):
    """C_target and the loss weight, from a record's analysis channels.

    Kept in one function because eval_rob.py and mine_rob.py must form the
    target exactly as training did; two copies of this arithmetic would drift
    and the drift would look like a model regression.
    """
    r_ideal = px[..., CH_IDEAL_R]
    r_norm = px[..., CH_RNORM]
    tgt = np.clip((r_ideal + TGT_EPS) / (r_norm + TGT_EPS), 0.0, 1.0)
    # Impact weighting: the error in the FINAL mask is r_norm * (C - target).
    w = px[..., CH_W] * np.maximum(r_norm, 0.02)
    return tgt.astype(np.float32), w.astype(np.float32)


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
                             gain=float(p[12]) if len(p) > 12 else 1.0,
                             phase=int(p[13]) if len(p) > 13 else 0))
    assert len(recs) == n, f"idx has {len(recs)} rows, meta says {n}"
    return data, meta, recs


def split_records(recs, guide_w=2016, holdout_burst=None):
    """Train/eval split.

    Two independent holdouts, because they answer different questions:

      * REGION. Crops whose origin lies in the right-hand strip of the frame
        are eval-only for every burst. This measures generalisation to unseen
        content within scenes the model has otherwise trained on.
      * BURST. Optionally an entire burst is withheld. A region holdout cannot
        answer "does this work on a scene it has never seen", since the two
        regions share a scene, an exposure and a noise level.

    Splitting on the crop ORIGIN rather than on the record index also keeps a
    training crop from overlapping an eval crop, which would leak. A paired
    record shares its origin with its corrupted twin, so the pair always lands
    on the same side of the split -- which it must, or the clean half of a pair
    would be scoring the model on content its twin trained on.
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

    Separable rather than dense because this runs per comparison frame over the
    whole guide plane, where cost is set by pixels, not parameters: a dense
    32-channel stack is 26.8k MAC/px -- 490 GMAC for a 6-frame burst. Factoring
    each 3x3 into depthwise + pointwise and halving the width gives ~1.6k
    MAC/px for the same receptive field. Width 16 is also a multiple of the ANE
    tile width, so the channel dimension packs without waste.

    The 7-pixel receptive field is not decoration here. Feature channels 20-26
    are PLANES of the reference luma, the warped comparison luma and their
    residual, so the convolution stack is what turns them into a 15x15 guide-
    pixel patch per output pixel -- and a 15x15 patch spanning most of two
    16-raw-pixel tiles is what lets the residual's RAMP ACROSS a tile be seen
    at all. That ramp is the signature of a translation-only flow fitting a
    locally rotating scene, and no pointwise statistic carries it.

    The output bias starts at +5.0, i.e. C = 0.993 everywhere before a single
    gradient step. "Leave the analytic mask alone" is the initialisation; the
    model has to earn every departure from it.
    """
    def __init__(self, cin=IN_CH, w=16, bias0=5.0):
        super().__init__()
        def block(d):
            return [nn.Conv2d(w, w, 3, padding=d, dilation=d, groups=w),
                    nn.Conv2d(w, w, 1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(
            nn.Conv2d(cin, w, 1), nn.ReLU(inplace=True),
            *block(1), *block(2), *block(4),
            nn.Conv2d(w, 1, 1),
        )
        head = self.net[-1]
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, bias0)

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
    npair = sum(1 for r in recs if r["phase"] == 1)
    print(f"  paired clean phases: {npair} of {len(recs)}")
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

    # ROB_DROP_CH removes input channels by zeroing them AFTER normalisation,
    # which is exactly equivalent to deleting them: the first 1x1 conv then
    # receives a constant 0 on those lanes and their weights cannot influence
    # anything. Done this way so an ablation needs no dataset regeneration --
    # in particular the controlled with/without match-quality comparison, where
    # both arms must otherwise be identical.
    drop_ch = [int(t) for t in os.environ.get("ROB_DROP_CH", "").split(",")
               if t.strip() != ""]
    if drop_ch:
        print(f"dropping input channels {drop_ch}")

    ps = int(os.environ.get("ROB_PATCH", min(96, gh, gw)))
    bs = int(os.environ.get("ROB_BATCH", 24))
    steps = int(os.environ.get("ROB_STEPS", 4000))

    def sample_batch():
        xs, ys, ws = [], [], []
        for _ in range(bs):
            r = train[rng.randint(len(train))]
            y0 = rng.randint(max(1, gh - ps))
            x0 = rng.randint(max(1, gw - ps))
            p = np.asarray(data[r["rec"], y0:y0 + ps, x0:x0 + ps, :], dtype=np.float32)
            tgt, w = target_and_weight(p)
            xs.append((p[..., :IN_CH] - mu) / sd)
            ys.append(tgt[..., None])
            ws.append(w[..., None])
        t = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2)
        x, y, w = t(xs), t(ys), t(ws)
        for c in drop_ch:
            x[:, c] = 0.0
        return x, y, w

    bce = nn.BCEWithLogitsLoss(reduction="none")

    # The cost is asymmetric and the two halves pull in opposite directions.
    #
    #   a merged misalignment is unacceptable  -> a missed correction costs FA_W
    #   a forgone merge is merely wasteful     -> an unneeded one costs 1
    #
    # but also, and this is the part a plain reweighting does not give: a
    # provably-safe pixel must reach ~1.0, not 0.98, because C multiplies. A
    # correction of 0.98 applied to the 95% of the frame that is perfectly
    # aligned is a 2% darkening of the entire mask -- invisible in any average
    # and exactly the "rescaled the mask" failure this model is not allowed to
    # have. Hence the hinges: they act on the LOGIT with constant gradient
    # until their margin is met, so unlike BCE they keep pushing at the top of
    # the range where the model would otherwise quietly drift down.
    FA_W = float(os.environ.get("ROB_FA_W", 6.0))
    SAFE_W = float(os.environ.get("ROB_SAFE_W", 4.0))
    M_POS = float(os.environ.get("ROB_M_POS", 6.0))    # logit 6 = 0.9975
    M_NEG = float(os.environ.get("ROB_M_NEG", -2.0))   # logit -2 = 0.12

    def loss_fn(logits, target, w):
        # Harm magnitude, recovered from the target. Weighting by it makes
        # visibly-wrong pixels dominate the objective and stops near-zero-harm
        # cases -- numerous, and invisible -- from consuming capacity.
        mag = (1.0 - target).clamp(0.0, 1.0)
        ww = w * (1.0 + FA_W * mag)
        l = (bce(logits, target) * ww).sum() / ww.sum().clamp_min(1e-6)

        safe = (target > 0.999).float() * w
        if float(safe.sum()) > 0:
            l = l + SAFE_W * ((M_POS - logits).clamp(min=0.0) * safe).sum() \
                    / safe.sum().clamp_min(1e-6)
        bad = (target < 0.2).float() * w
        if float(bad.sum()) > 0:
            l = l + FA_W * ((logits - M_NEG).clamp(min=0.0) * bad).sum() \
                    / bad.sum().clamp_min(1e-6)
        return l

    lr0 = float(os.environ.get("ROB_LR", 3e-3))
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    for it in range(steps):
        for g in opt.param_groups:
            # Cosine decay to zero. What is being fitted is a VALUE the merge
            # multiplies by, not a ranking; holding lr flat to the last step
            # leaves the weights bouncing around the minimum, which shows up as
            # a squashed output range.
            g["lr"] = lr0 * 0.5 * (1.0 + np.cos(np.pi * it / max(1, steps - 1)))
        x, y, w = sample_batch()
        loss = loss_fn(model.logits(x), y, w)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, steps // 15) == 0 or it == steps - 1:
            print(f"  step {it:5d}  loss {loss.item():.5f}")

    ckpt_path = os.environ.get("ROB_OUT", os.path.join(SC, "robnet.pt"))
    # drop_ch travels WITH the weights. If evaluation forgot to apply it the
    # model would be fed a channel it never trained on, which looks like a
    # mysterious accuracy collapse rather than a harness mistake.
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd,
                "in_ch": IN_CH, "holdout_burst": holdout_burst,
                "drop_ch": drop_ch}, ckpt_path)
    with open(os.path.splitext(ckpt_path)[0] + "_norm.json", "w") as f:
        json.dump({"mu": mu.tolist(), "sd": sd.tolist(), "in_ch": IN_CH}, f)
    print("\nsaved", ckpt_path)
    print("run eval_rob.py for the two numbers that decide this")


if __name__ == "__main__":
    main()
