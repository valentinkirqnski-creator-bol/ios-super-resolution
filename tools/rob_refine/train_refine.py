"""Train the robustness REFINEMENT network.

The network predicts one number per pixel, q_keep in [0,1], and the pipeline
uses it only as

    R_final = R_wronski * (1 - kappa * (1 - q_keep))

so it can never raise R and never resurrect a pixel Eq. 5-9 rejected. Its
target is

    q_keep* = clamp(R* / R_wronski, 0, 1)

the multiplier that would turn the analytic mask into the ideal one, where R*
is the inverse-MSE optimal merge weight derived from ground-truth motion in
refine_dataset.cpp. Clamping at 1 is what makes the whole stage one-sided:
wherever the analytic mask is already at or below the ideal weight the target
is exactly 1 and the correct output is "change nothing".

Two things in here are worth reading before changing anything.

Sampling is stratified, not uniform. Measured on this data, about a third of
all the mask's correctable error sits at pixels whose flow is wrong by more
than 5 raw pixels -- gross search failures over flat content, which are the
REPLACEMENT network's problem (tools/rob_nn) and not this one. Train on the
natural distribution and that mass sets the weights. The strata below force
the sub-pixel regime this stage exists for to carry its own share, and cap
what the gross regime can contribute.

The loss is asymmetric twice over, deliberately. L_fp/L_fn implement the
stated preference -- a wrongly rejected pixel costs detail that cannot be
recovered, a missed artifact costs one subtly doubled edge -- and are dialled
past what physics alone would ask for. L_merge is the physics: the actual
excess mean squared error of the merged pixel, which for a reference of weight
1 and a comparison sample of weight w is

    MSE(w) / sigma^2 = [1 + w^2 / R*] / (1 + w)^2

minimised at exactly w = R*. Reported relative to that minimum it is zero when
the weight is right and grows in the correct, asymmetric way on both sides:
too high pays the misalignment, too low pays the noise that was not averaged
away. That term is what makes the objective "will the merged pixel be worse"
rather than "is the mask's number close to another number". It is also
computed through the cap and the multiply the pipeline will actually apply,
not on q directly -- train on q alone and the network optimises a mask nothing
downstream ever uses.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn

SC = os.path.dirname(os.path.abspath(__file__))
PREFIX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SC, "refineset")

# ---- layout, mirroring refine_dataset.cpp -------------------------------
NAMES = ["R", "z_d", "tex", "s", "Ex", "Ey", "Emag", "div", "curl", "Mspan",
         "gx", "gy", "gmag", "E_perp", "E_par", "coh", "lap", "res", "res_disp",
         "res_z", "bright", "nsig", "agree", "mismatch",
         # how much of sigma^2 is the scene's own texture
         "sig_ms_frac",
         # noise-aware edge detection, then the pooled displacement
         "edge_snr", "aniso", "delta_lk", "sd_lk", "t_lk",
         # where in the tile, and whether the geometry agrees with the
         # measurement once both are in units of its uncertainty
         "u_n", "v_n", "agree_lk"]
IN_CH = len(NAMES)                 # 36
# Derived, not hardcoded: they moved when the feature count changed, and a
# stale literal here reads the target out of a feature column.
CH_RGEOM = IN_CH
CH_RSTAR = IN_CH + 1
CH_FLOWERR = IN_CH + 2
CH_DELTA = IN_CH + 3
CH_SIGMA = IN_CH + 4

# Channels with a heavy tail: the flow-derived lengths reach hundreds of
# pixels over flat content where the block search wandered, while the regime
# that matters is fractions of a pixel. A signed log keeps both ends
# representable in one normalised input instead of letting the tail set the
# scale and crush everything real into a rounding error. Folded into the
# exported graph, so nothing outside these weights has to know about it.
LOG_CH = [1, 4, 5, 6, 7, 8, 9, 13, 14, 16, 17, 18, 19, 22, 23,
          27, 28, 29, 32]

KAPPA = float(os.environ.get("ROB_REFINE_KAPPA", 0.75))
ARCH = os.environ.get("ROB_REFINE_ARCH", "mlp")             # mlp | cnn
BASELINE = os.environ.get("ROB_REFINE_BASELINE", "geom")    # plain | geom
LAM_FP = float(os.environ.get("ROB_REFINE_LAM_FP", 8.0))
LAM_FN = float(os.environ.get("ROB_REFINE_LAM_FN", 1.0))
LAM_MERGE = float(os.environ.get("ROB_REFINE_LAM_MERGE", 1.0))
LAM_ID = float(os.environ.get("ROB_REFINE_LAM_ID", 0.02))
STEPS = int(os.environ.get("ROB_REFINE_STEPS", 3000))
HOLDOUT = int(os.environ.get("ROB_REFINE_HOLDOUT", 8))
WIDTH = int(os.environ.get("ROB_REFINE_WIDTH", 16))
SEED = int(os.environ.get("ROB_REFINE_SEED", 0))
POOL_PER_FRAME = int(os.environ.get("ROB_REFINE_POOL", 40000))
# Significance gate on the TARGET, in units of excess merged-pixel MSE.
#
# The inverse-MSE optimal weight R* is slightly below 1 almost everywhere,
# because almost every pixel carries some residual misalignment. Followed
# literally the target would therefore ask the network to move ~96% of the
# frame -- technically optimal, and the opposite of the sparse correction this
# stage is supposed to be. Worse, a uniform small reduction applied to every
# comparison frame is very nearly INERT anyway: the merge normalises num/den,
# so only the differences between frames survive.
#
# So the target is pinned to exactly 1 wherever leaving R alone costs less
# than this much extra mean squared error in the merged pixel. 0.02 is 2%
# excess MSE, about 1% in amplitude -- below anything visible. The effect is
# that the network is trained to say "change nothing" on the vast majority of
# pixels and its output means something where it does not.
GATE = float(os.environ.get("ROB_REFINE_GATE", 0.02))

# Split by true flow error: that is what separates the regime this stage is
# for from the regime the replacement network is for. Each stratum gets a
# fixed share of every batch regardless of how rare it is in the data.
STRATA = [
    ("aligned   <0.1px", 0.0, 0.1, 0.30),   # must NOT be rejected: the guard
    ("subpixel .1-.5px", 0.1, 0.5, 0.30),   # the target regime
    ("near      .5-1px", 0.5, 1.0, 0.20),
    ("moderate   1-3px", 1.0, 3.0, 0.15),
    ("gross       >3px", 3.0, 1e9, 0.05),   # deliberately a small share
]


def load(prefix):
    meta = {}
    for line in open(prefix + ".meta"):
        k, v = line.split()
        meta[k] = int(v)
    H, W, C, NF = meta["height"], meta["width"], meta["channels"], meta["frames"]
    d = np.memmap(prefix + ".f32", dtype=np.float32, mode="r", shape=(NF, H, W, C))
    return d, H, W, C, NF


def baseline_R(px, baseline=BASELINE):
    """The analytic mask this run refines. Channel 0 is it with
    motion_geom_reject off, channel 24 with it on; no other feature channel
    depends on that toggle, which is why one dataset serves both."""
    return px[..., CH_RGEOM] if baseline == "geom" else px[..., 0]


def features(px, baseline=BASELINE):
    """Feature tensor for the selected baseline: channel 0 carries whichever
    analytic mask is being refined."""
    f = np.array(px[..., :IN_CH], dtype=np.float32)
    if baseline == "geom":
        f[..., 0] = px[..., CH_RGEOM]
    return f


def merge_excess_np(w, rstar):
    """Excess merged-pixel MSE relative to its own minimum; see merge_excess."""
    rs = np.clip(rstar, 1e-3, 1.0)
    return ((1.0 + w * w / rs) / (1.0 + w) ** 2) * (1.0 + rs) - 1.0


def target_q(px, baseline=BASELINE):
    R = baseline_R(px, baseline)
    rstar = px[..., CH_RSTAR]
    q = np.clip(rstar / np.maximum(R, 1e-6), 0.0, 1.0)
    # The significance gate: where leaving R alone costs less than GATE of
    # excess merged MSE, the right answer is exactly "change nothing".
    q = np.where(merge_excess_np(R, rstar) < GATE, 1.0, q).astype(np.float32)
    return R, rstar, q


def finite(px):
    """R itself can be NaN where the flow leaves the frame (Eq. 6 evaluates
    inf/inf there). Those pixels are already lost to the merge and must not be
    allowed to set a gradient -- nor to enter any average."""
    ok = np.isfinite(px[..., :IN_CH]).all(-1)
    ok &= np.isfinite(px[..., CH_RGEOM]) & np.isfinite(px[..., CH_RSTAR])
    ok &= np.isfinite(px[..., CH_FLOWERR])
    ok &= np.isfinite(px[..., 0])
    return ok


def usable(px, baseline=BASELINE, require_live=True):
    """Finite, and -- for TRAINING only -- a pixel the analytic mask still
    trusts. R below 0.02 has effectively been rejected already: there is
    nothing to refine, and dividing by it to form the target manufactures
    noise.

    require_live must be False when EVALUATING. Applying it there silently
    drops every pixel the geometry rejection zeroed, which is precisely where
    configurations 1 and 2 differ -- with it on, the two scored identically
    and the geometry test looked inert when it was not."""
    ok = finite(px)
    if require_live:
        ok &= baseline_R(px, baseline) > 0.02
    return ok


def signed_log_np(x, ch=LOG_CH):
    out = np.array(x, dtype=np.float32, copy=True)
    out[..., ch] = np.sign(out[..., ch]) * np.log1p(np.abs(out[..., ch]))
    return out


def signed_log_t(x, ch=LOG_CH):
    out = x.clone()
    out[:, ch] = torch.sign(x[:, ch]) * torch.log1p(torch.abs(x[:, ch]))
    return out


class RefineMLP(nn.Module):
    """Pointwise. Every feature that needs a neighbourhood -- the structure
    tensor, the flow-field gradient, the residual's local mean -- is already
    computed in build_robustness_refine_features, so the network itself sees
    one pixel at a time.

    That is not a compromise, it is the point. A receptive field of one pixel
    means the strip decomposition on device is exactly equivalent to
    whole-plane inference, with no halo and no possibility of a seam; the
    activations are WIDTH floats per pixel rather than a 32-channel tensor;
    and the whole thing is under a thousand multiply-accumulates per pixel.
    Try this first and only reach for the CNN if it measurably cannot do the
    job."""

    def __init__(self, cin=IN_CH, w=WIDTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, w, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, 1, 1),
        )
        # Start at q_keep ~ 0.95, i.e. at the identity. Not cosmetic: from a
        # 0.5 start the network spends its first few hundred steps rejecting
        # half the frame, and with this loss the false-positive term is steep
        # enough there to push it into a corner it does not come back from.
        nn.init.constant_(self.net[-1].bias, 3.0)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class RefineCNN(nn.Module):
    """5x5 receptive field (two depthwise-separable 3x3s), for the case where
    the pointwise model measurably cannot separate a doubled edge from a sharp
    one. Shipping this means raising kRobustnessRefineHalo to 2 to match, or
    the strips seam."""

    def __init__(self, cin=IN_CH, w=WIDTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, w, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1, groups=w), nn.Conv2d(w, w, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1, groups=w), nn.Conv2d(w, w, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, 1, 1),
        )
        nn.init.constant_(self.net[-1].bias, 3.0)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def merge_excess(w, rstar):
    """Excess merged-pixel MSE, relative to its own minimum, for a comparison
    sample of weight w against a reference of weight 1:

        MSE(w)/sigma^2 = [1 + w^2 / R*] / (1 + w)^2,  minimised at w = R*

    Zero when the weight is right, and growing on both sides in the way the
    picture actually pays for -- too high pays the misalignment, too low pays
    the noise that was not averaged away."""
    rs = torch.clamp(rstar, 1e-3, 1.0)
    mse = (1.0 + w * w / rs) / (1.0 + w) ** 2
    best = 1.0 / (1.0 + rs)
    return mse / best - 1.0


def build_pools(data, n_train, rng, baseline=BASELINE):
    """Stratified pixel pools, materialised in RAM.

    Drawing single pixels out of the memmap inside the training loop is a few
    million random reads per run and dominates everything else; the pools are
    a couple of hundred MB and make a batch a fancy-index."""
    pools = {name: [] for name, _, _, _ in STRATA}
    for f in range(n_train):
        px = np.asarray(data[f], dtype=np.float32)
        ok = usable(px, baseline)
        fe = px[..., CH_FLOWERR]
        for name, lo, hi, _share in STRATA:
            sel = ok & (fe >= lo) & (fe < hi)
            ys, xs = np.nonzero(sel)
            if ys.size == 0:
                continue
            take = min(ys.size, POOL_PER_FRAME)
            pick = rng.choice(ys.size, take, replace=False)
            pools[name].append(px[ys[pick], xs[pick], :])
    for k in list(pools):
        pools[k] = (np.concatenate(pools[k], 0) if pools[k]
                    else np.zeros((0, data.shape[-1]), np.float32))
    return pools


def save_host_bin(model, mu, sd, path):
    """Plain little-endian dump of the same weights, for refine_bench.cpp.

    Core ML exists only on Apple platforms, so the only way to exercise the
    SHIPPED C++ path -- build_robustness_refine_features, the strip loop, the
    bounded multiply -- anywhere else is to hand it an evaluator that reads
    this. It is a test fixture, not a deployment format: the app loads the
    Core ML model that export_coreml.py writes from the same checkpoint."""
    import struct
    ws = [p.detach().numpy().astype(np.float32).ravel()
          for p in model.net.parameters()]
    with open(path, "wb") as f:
        f.write(b"RFN1")
        f.write(struct.pack("<iiii", 0 if ARCH == "mlp" else 1, IN_CH, WIDTH,
                            len(LOG_CH)))
        f.write(np.asarray(LOG_CH, np.int32).tobytes())
        f.write(np.asarray(mu, np.float32).tobytes())
        f.write(np.asarray(sd, np.float32).tobytes())
        f.write(struct.pack("<i", len(ws)))
        for w in ws:
            f.write(struct.pack("<i", w.size))
            f.write(w.tobytes())
    print(f"saved {path} ({sum(w.size for w in ws)} weights, for refine_bench)")


def main():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    data, H, W, C, NF = load(PREFIX)
    n_train = max(1, NF - HOLDOUT)
    print(f"dataset {PREFIX}: {NF} frames {H}x{W}x{C}  "
          f"({n_train} train, {NF - n_train} held out)")
    print(f"baseline={BASELINE} arch={ARCH} kappa={KAPPA} gate={GATE} "
          f"lam_fp={LAM_FP} lam_fn={LAM_FN} lam_merge={LAM_MERGE} "
          f"lam_id={LAM_ID} steps={STEPS}")

    pools = build_pools(data, n_train, rng)
    live_strata = []
    print()
    for name, lo, hi, share in STRATA:
        n = pools[name].shape[0]
        print(f"  stratum {name:<18} {n:>9} pixels   batch share {share:.0%}")
        if n:
            live_strata.append((name, share))
    if not live_strata:
        raise SystemExit("no usable pixels -- check the generator output")
    tot = sum(s for _, s in live_strata)
    live_strata = [(n, s / tot) for n, s in live_strata]

    # ---- normalisation, on the stratified mixture rather than a raw frame,
    # so the statistics describe what the network is actually shown.
    sample = np.concatenate([
        features(pools[n][rng.choice(pools[n].shape[0],
                                     min(pools[n].shape[0], 60000), replace=False)])
        for n, _ in live_strata], 0)
    sample = signed_log_np(sample)
    mu = sample.mean(0).astype(np.float32)
    sd = (sample.std(0) + 1e-6).astype(np.float32)
    print("\nnormalised inputs (after the signed log where it applies):")
    for c in range(IN_CH):
        print(f"  {c:>2} {NAMES[c]:<10} mean {mu[c]:+10.4f}  std {sd[c]:10.4f}"
              + ("   [log]" if c in LOG_CH else ""))
    mu_t = torch.from_numpy(mu).view(1, IN_CH, 1, 1)
    sd_t = torch.from_numpy(sd).view(1, IN_CH, 1, 1)

    # What the gate actually leaves for the network to do. Printed because it
    # is the number that decides whether this is a sparse correction or a new
    # mask, and it should be read before any of the loss values below.
    probe = np.concatenate([pools[n][:20000] for n, _ in live_strata], 0)
    _R, _Rs, qt = target_q(probe)
    print(f"\ntarget after the gate, over a stratified sample:")
    print(f"  exactly 1.00 (change nothing)  {(qt >= 0.9999).mean()*100:5.1f}%")
    for lo, hi in ((0.9, 0.9999), (0.5, 0.9), (0.1, 0.5), (0.0, 0.1)):
        print(f"  {lo:4.2f} .. {hi:<6.4g}                {((qt > lo) & (qt <= hi)).mean()*100:5.1f}%")

    def draw(bs=16384):
        rows = []
        for name, share in live_strata:
            pool = pools[name]
            n = max(1, int(bs * share))
            rows.append(pool[rng.randint(0, pool.shape[0], n)])
        px = np.concatenate(rows, 0)
        R, _Rs, q = target_q(px)
        f = torch.from_numpy(features(px)).view(-1, IN_CH, 1, 1)
        return ((signed_log_t(f) - mu_t) / sd_t,
                torch.from_numpy(q).view(-1, 1, 1, 1),
                torch.from_numpy(R).view(-1, 1, 1, 1),
                torch.from_numpy(np.asarray(px[..., CH_RSTAR])).view(-1, 1, 1, 1))

    def draw_patches(bs=24, ps=48):
        """Contiguous tiles for the convolutional variant, centred on frames
        drawn at random. Patches cannot be stratified per pixel, so the CNN
        sees the natural distribution inside each tile -- one more reason to
        prefer the pointwise model unless it demonstrably falls short."""
        fs, qs, rs, ss = [], [], [], []
        for _ in range(bs):
            f = rng.randint(n_train)
            y0 = rng.randint(H - ps); x0 = rng.randint(W - ps)
            px = np.asarray(data[f, y0:y0 + ps, x0:x0 + ps, :], dtype=np.float32)
            R, Rs, q = target_q(px)
            ok = usable(px)
            fs.append(np.nan_to_num(features(px)))
            qs.append(q); rs.append(np.where(ok, R, 0.0)); ss.append(np.nan_to_num(Rs))
        f = torch.from_numpy(np.stack(fs)).permute(0, 3, 1, 2)
        return ((signed_log_t(f) - mu_t) / sd_t,
                torch.from_numpy(np.stack(qs)).unsqueeze(1),
                torch.from_numpy(np.stack(rs)).unsqueeze(1),
                torch.from_numpy(np.stack(ss)).unsqueeze(1))

    model = RefineMLP() if ARCH == "mlp" else RefineCNN()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"\nmodel {ARCH}: {nparam} parameters, "
          f"{nparam - 2 * WIDTH - 1} MAC/pixel approx")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

    for it in range(STEPS):
        x, t, r, rstar = draw() if ARCH == "mlp" else draw_patches()
        p = model(x)
        live = (r > 0.02).float()
        n = live.sum().clamp(min=1.0)

        over = torch.relu(t - p)     # predicted keep below ideal: over-rejection
        under = torch.relu(p - t)    # predicted keep above ideal: missed artifact
        l_fp = (live * over ** 2).sum() / n
        l_fn = (live * under ** 2).sum() / n

        r_final = r * (1.0 - KAPPA * (1.0 - p))
        l_merge = (live * merge_excess(r_final, rstar)).sum() / n
        # Standing cost on any deviation from "keep", independent of the
        # target, so a correction has to be worth making at all.
        l_id = (live * (1.0 - p)).sum() / n

        loss = LAM_FP * l_fp + LAM_FN * l_fn + LAM_MERGE * l_merge + LAM_ID * l_id
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 200 == 0 or it == STEPS - 1:
            print(f"  step {it:5d}  loss {loss.item():.5f}   fp {l_fp.item():.5f}"
                  f"  fn {l_fn.item():.5f}  merge {l_merge.item():.5f}"
                  f"  id {l_id.item():.4f}   mean q {p.mean().item():.4f}")

    out = os.path.join(SC, f"refinenet_{ARCH}_{BASELINE}.pt")
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd,
                "arch": ARCH, "baseline": BASELINE, "in_ch": IN_CH,
                "log_ch": LOG_CH, "width": WIDTH, "kappa": KAPPA}, out)
    save_host_bin(model, mu, sd, os.path.splitext(out)[0] + ".bin")
    with open(os.path.join(SC, "refinenet_norm.json"), "w") as f:
        json.dump({"mu": mu.tolist(), "sd": sd.tolist(), "in_ch": IN_CH,
                   "log_ch": LOG_CH, "names": NAMES}, f, indent=1)
    print(f"\nsaved {out}")
    print("now run eval_refine.py for the four-way comparison on held-out frames")


if __name__ == "__main__":
    main()
