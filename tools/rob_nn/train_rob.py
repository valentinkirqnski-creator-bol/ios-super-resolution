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
consume capacity -- so below R_MIN_TRAIN it gets no gradient at all, not
merely a small one. The samples that are worth anything are the quadrant

    R_normal HIGH  and  R_ideal LOW

which is the definition of an analytic-mask failure and the entire reason this
model exists. It is a fraction of a percent of pixels, so patches are drawn
towards it rather than uniformly, and the fraction is printed.

Two things this training deliberately does NOT do
-------------------------------------------------
It does not trust the baseline flow just because the real aligner produced it.
harm() is measured against flow_base, so where flow_base is itself wrong the
supervision is wrong -- and that is worst exactly during rotation and complex
motion, the cases this model exists for. fb_valid.py carries an INDEPENDENT
forward-backward check and its verdict zeroes the weight. Nothing here calls
the label "exact".

It does not treat feature channel 25 as a measurement of anything. That
channel is r*|g| / (|g|^2 + eps), which is first-order brightness constancy,
and brightness constancy is violated by rotation, interpolation, aliasing,
clipping, Bayer phase and any displacement that is not small -- i.e. by most of
the cases of interest. It is an input the convolution may exploit and it
appears in no label, no weight and no metric.

The failure mode to design against
----------------------------------
A model that lowers C everywhere would score well on detection and be
worthless: it has rescaled the mask, not corrected it. Three things guard
against that, and the third is the one that actually decides it:

  1. the ratio target is exactly 1.0 wherever the analytic mask is not too
     high, which is the overwhelming majority of pixels;
  2. the output bias starts at +8 (C = 0.99966), so "do nothing" is the
     initialisation and departures from it have to be earned;
  3. a hinge that keeps the logit above M_POS = 8 (C = 0.99966) on
     provably-safe pixels, with constant gradient. BCE's gradient is
     (p - target) and fades to nothing exactly where the model is drifting from
     0.999 to 0.98 -- a drift that is invisible in the loss and very visible in
     the mask, because C multiplies every pixel of the frame.

Reported at the end: the distribution of C on correctly-aligned pixels. If its
mean is not within a hair of 1, the model is a failure regardless of what its
detection numbers say.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fb_valid

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
# ... and below this it gets no gradient at all. C cannot make a pixel the
# closed form has already discarded matter, so spending capacity there is
# spending it on nothing.
R_MIN_TRAIN = float(os.environ.get("ROB_R_MIN", 0.2))


def target_and_weight(px, fb_ok=None):
    """C_target and the loss weight, from a record's analysis channels.

    Kept in one function because eval_rob.py and mine_rob.py must form the
    target exactly as training did; two copies of this arithmetic would drift
    and the drift would look like a model regression.

    fb_ok, when given, is the independent forward-backward verdict on the
    baseline flow for each pixel (fb_valid.py). Where it is False the harm
    label was measured against a correspondence nothing has vouched for, so the
    pixel contributes no gradient -- it is unknown, not safe.
    """
    r_ideal = px[..., CH_IDEAL_R]
    r_norm = px[..., CH_RNORM]
    # A NaN R_normal is possible on data generated before the analytic mask's
    # 0/0 guard (compute_robustness_analytic); treat those pixels as unknown
    # rather than letting one of them turn a whole batch into NaN. Substituted
    # BEFORE the arithmetic, not after, or the multiply warns and the NaN has
    # already propagated into tgt.
    ok = np.isfinite(r_norm) & np.isfinite(r_ideal)
    r_norm = np.where(ok, r_norm, 0.0)
    r_ideal = np.where(ok, r_ideal, 0.0)
    tgt = np.clip((r_ideal + TGT_EPS) / (r_norm + TGT_EPS), 0.0, 1.0)
    # Impact weighting: the error in the FINAL mask is r_norm * (C - target),
    # and it is exactly zero where the analytic mask has already rejected.
    w = px[..., CH_W] * np.where(r_norm >= R_MIN_TRAIN, r_norm, 0.0) * ok
    if fb_ok is not None:
        w = w * fb_ok
    return tgt.astype(np.float32), w.astype(np.float32)


def quadrant(px):
    """The analytic mask's own failures: it trusts, and merging harms."""
    return ((px[..., CH_W] > 0) & (px[..., CH_RNORM] >= 0.5) &
            (px[..., CH_IDEAL_R] < 0.5))


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

    The output bias starts at +8.0, i.e. C = 0.99966 everywhere before a
    single gradient step, with the head weights zeroed so the initial output is
    exactly constant. "Leave the analytic mask alone" is the initialisation and
    the model has to earn every departure from it. Starting at +5 was tried
    first and is not enough: 0.993 over a whole frame is already a 0.7%
    darkening of the mask, which is the rescaling failure in miniature.
    """
    def __init__(self, cin=IN_CH, w=16, bias0=8.0):
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

    # ---- the independent verdict on the baseline flow ----------------------
    fb = fb_valid.load_fb(os.environ.get("ROB_FB", os.path.join(SC, "fb")))
    print("\n" + fb_valid.summarise(fb))
    if not fb:
        print("  WARNING: training WITHOUT it. Every harm label is then only as")
        print("  good as the aligner that produced the baseline, and that is")
        print("  weakest exactly during rotation, which is what this model is for.")

    def load_record(rec_meta, y0, x0, h, w_):
        p = np.asarray(data[rec_meta["rec"], y0:y0 + h, x0:x0 + w_, :],
                       dtype=np.float32)
        sub = dict(rec_meta); sub["cy"] = rec_meta["cy"] + y0; sub["cx"] = rec_meta["cx"] + x0
        mag = np.hypot(p[..., 9], p[..., 10])
        fb_ok, rot = fb_valid.record_tile_maps(fb, sub, p.shape[0], p.shape[1], mag)
        return p, fb_ok, rot

    # ---- per-record sampling priority, updated during training -------------
    #
    # Uniform sampling spends almost all of its time on records that are
    # already right: the quadrant this model exists for is a fraction of a
    # percent of pixels. Two things are boosted, and the SECOND matters as much
    # as the first:
    #
    #   * records where a harmful pixel the analytic mask trusts is still being
    #     merged -- the misses;
    #   * records where CORRECT ROTATION is being pulled below 1 -- hard
    #     positives. Paired examples alone do not prevent the network learning
    #     a subtle proxy for "large or non-uniform flow is suspicious", because
    #     that proxy is right often enough to survive an unweighted average.
    #     Mining them explicitly is what closes it, and rotation is identified
    #     from the MEASURED global rotation (fb_check.cpp), not from flow
    #     magnitude or flow spread, which parallax also produces.
    prio = np.ones(len(train), np.float64)
    mine_every = int(os.environ.get("ROB_MINE_EVERY", 400))
    mine_n = int(os.environ.get("ROB_MINE_BATCH", 60))
    ROT_BOOST = float(os.environ.get("ROB_ROT_BOOST", 8.0))

    def remine():
        idx = rng.choice(len(train), size=min(mine_n, len(train)), replace=False)
        for i in idx:
            r = train[i]
            p, fb_ok, rot = load_record(r, 0, 0, gh, gw)
            tgt, w = target_and_weight(p, fb_ok)
            if w.sum() <= 0:
                prio[i] = 0.05
                continue
            with torch.no_grad():
                x = torch.from_numpy(((p[..., :IN_CH] - mu) / sd)).permute(2, 0, 1)[None]
                for c in drop_ch:
                    x[:, c] = 0.0
                c_hat = model(x)[0, 0].numpy()
            miss = ((w > 0) & (tgt < 0.5) & (c_hat > 0.5)).sum()
            n_bad = max(int(((w > 0) & (tgt < 0.5)).sum()), 1)
            safe = (w > 0) & (tgt > 0.999)
            if rot is not None:
                rot_safe = safe & (rot > fb_valid.ROT_STRONG)
            else:
                rot_safe = safe
            n_rs = max(int(rot_safe.sum()), 1)
            hurt_rot = int((rot_safe & (c_hat < 0.99)).sum())
            prio[i] = 0.25 + miss / n_bad + ROT_BOOST * hurt_rot / n_rs

    def sample_batch():
        xs, ys, ws = [], [], []
        pr = prio / prio.sum()
        for _ in range(bs):
            r = train[rng.choice(len(train), p=pr)]
            # Aim the patch at the quadrant when the record has one, so the
            # 0.5%-of-pixels signal is not diluted to nothing by the crop.
            y0 = rng.randint(max(1, gh - ps))
            x0 = rng.randint(max(1, gw - ps))
            if rng.rand() < 0.6:
                full = np.asarray(data[r["rec"], :, :, :], dtype=np.float32)
                q = np.nonzero(quadrant(full))
                if len(q[0]):
                    k = rng.randint(len(q[0]))
                    y0 = int(np.clip(q[0][k] - ps // 2, 0, max(0, gh - ps)))
                    x0 = int(np.clip(q[1][k] - ps // 2, 0, max(0, gw - ps)))
                del full
            p, fb_ok, _ = load_record(r, y0, x0, ps, ps)
            tgt, w = target_and_weight(p, fb_ok)
            xs.append((p[..., :IN_CH] - mu) / sd)
            ys.append(tgt[..., None])
            ws.append(w[..., None])
        t = lambda a: torch.from_numpy(np.stack(a)).permute(0, 3, 1, 2)
        x, y, w = t(xs), t(ys), t(ws)
        for c in drop_ch:
            x[:, c] = 0.0
        return x, y, w

    # What fraction of the training set is actually the thing being fixed.
    qn = qd = 0
    for r in train[::max(1, len(train) // 120)]:
        p, fb_ok, _ = load_record(r, 0, 0, gh, gw)
        _, w = target_and_weight(p, fb_ok)
        qn += int((quadrant(p) & (w > 0)).sum()); qd += int((w > 0).sum())
    print(f"trainable pixels after validity + forward-backward + R_normal >= "
          f"{R_MIN_TRAIN}: {qd}")
    print(f"  of which the quadrant (R_normal >= 0.5 and R_ideal < 0.5): "
          f"{100.0 * qn / max(qd, 1):.3f}%  -- oversampled, not left at this rate")

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
    M_POS = float(os.environ.get("ROB_M_POS", 8.0))    # logit 8 = 0.99966
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
        if it % mine_every == 0:
            remine()
        x, y, w = sample_batch()
        loss = loss_fn(model.logits(x), y, w)
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {it}; check the dataset "
                             "for NaN in R_normal (analytic mask 0/0)")
        opt.zero_grad(); loss.backward(); opt.step()
        if it % max(1, steps // 15) == 0 or it == steps - 1:
            print(f"  step {it:5d}  loss {loss.item():.5f}  "
                  f"prio max {prio.max():.2f}")

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
