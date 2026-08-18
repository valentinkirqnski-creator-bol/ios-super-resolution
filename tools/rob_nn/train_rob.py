"""Train a robustness mask that predicts merge-safety directly.

Inputs are what any mask could see (reference stats, warped comparison stats,
the estimated flow and its local span, expected noise). The target is the
ideal robustness derived from ground truth: 1 where the alignment fetched the
right content, falling off where it did not. See rob_dataset.cpp for why that
label is not circular.

Also scores the classical Wronski mask against the same ground truth, which
is the first apples-to-apples number either has had.
"""
import json, os, sys
import numpy as np
import torch
import torch.nn as nn

SC = os.path.dirname(os.path.abspath(__file__))
PREFIX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SC, "robset2")
NCH, GH, GW = 16, 1512, 2016
IN_CH = 13          # channels 0..12 are inputs; 13 = harm, 14 = ideal R

meta = {}
with open(PREFIX + ".meta") as f:
    for line in f:
        k, v = line.split()
        meta[k] = int(v)
n_pixels = meta["pixels"]
n_frames = n_pixels // (GH * GW)
data = np.memmap(PREFIX + ".f32", dtype=np.float32, mode="r",
                 shape=(n_frames, GH, GW, NCH))
N_HOLD = int(os.environ.get("ROB_HOLDOUT", 6))
n_train = max(1, n_frames - N_HOLD)
print(f"dataset: {n_frames} frames x {GH}x{GW} x {NCH}ch "
      f"({n_train} train, {n_frames - n_train} held out)")

# ---------------------------------------------------------------- normalisation
# Flow is in raw pixels and can reach the hundreds; the statistics are in
# [0,1]. Without this the flow channels would dominate the first layer purely
# by scale. Constants are baked into the exported model so inference needs no
# external table.
sub = np.asarray(data[:n_train:max(1, n_train // 4), ::16, ::16, :], dtype=np.float32)
mu = sub[..., :IN_CH].reshape(-1, IN_CH).mean(0)
sd = sub[..., :IN_CH].reshape(-1, IN_CH).std(0) + 1e-6
print("input mean:", np.array2string(mu, precision=3))
print("input std :", np.array2string(sd, precision=3))

# ------------------------------------------------------- classical mask baseline
def classical_R(px, s1=2.0, s2=12.0, t=0.12, Mt=0.8):
    """Wronski Eq. 5-8 recomputed from the stored channels."""
    ref_m, ref_s = px[..., 0:3], px[..., 3:6]
    comp_m, nsig = px[..., 6:9], px[..., 12]
    d_ms_sq = ((ref_m - comp_m) ** 2).sum(-1)
    sig_ms_sq = (ref_s ** 2).sum(-1)
    # Guide has 3 channels, each with the modelled noise floor.
    nsq = 3.0 * nsig ** 2
    sig_sq = np.maximum(sig_ms_sq, nsq)
    shrink = d_ms_sq / np.maximum(d_ms_sq + nsq, 1e-12)
    d_sq = d_ms_sq * shrink ** 2
    s = np.where(px[..., 11] > Mt, s1, s2)
    return np.clip(s * np.exp(-d_sq / np.maximum(sig_sq, 1e-12)) - t, 0.0, 1.0)

# ---------------------------------------------------------------------- model
class RobNet(nn.Module):
    """Fully convolutional, dilated so the receptive field reaches ~30 raw px.

    The classical mask is a pointwise function of a 3x3 statistic, which is
    why it cannot tell a tile that disagrees with its neighbours from one that
    does not. Dilation is the cheapest way to give the decision that context
    without a resolution pyramid.
    """
    def __init__(self, cin=IN_CH, w=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, w, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=4, dilation=4), nn.ReLU(inplace=True),
            nn.Conv2d(w, w // 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(w // 2, 1, 1),
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x))

def sample_batch(bs=16, ps=96, rng=None):
    rng = rng or np.random
    xs, ys = [], []
    for _ in range(bs):
        f = rng.randint(n_train)
        y0 = rng.randint(GH - ps); x0 = rng.randint(GW - ps)
        p = np.asarray(data[f, y0:y0 + ps, x0:x0 + ps, :], dtype=np.float32)
        xs.append((p[..., :IN_CH] - mu) / sd)
        ys.append(p[..., 14:15])
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2)
    y = torch.from_numpy(np.stack(ys)).permute(0, 3, 1, 2)
    return x, y

def weighted_loss(pred, target):
    """Rejection is rare (a few percent of pixels), so an unweighted loss is
    minimised by predicting 1 everywhere -- exactly the failure being fixed."""
    w = 1.0 + 8.0 * (1.0 - target)
    return (w * (pred - target) ** 2).mean()

if __name__ == "__main__":
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    model = RobNet()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model: {nparam} parameters")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    steps = int(os.environ.get("ROB_STEPS", 400))
    for it in range(steps):
        x, y = sample_batch(rng=rng)
        loss = weighted_loss(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0 or it == steps - 1:
            print(f"  step {it:4d}  loss {loss.item():.5f}")

    # ------------------------------------------------------------- evaluation
    # Held-out frame, full field, both masks scored against ground truth.
    ev = np.asarray(data[n_frames - 1, ::2, ::2, :], dtype=np.float32)  # held out
    ideal = ev[..., 14]
    bad = ideal < 0.5              # ground truth: merging here does damage
    print(f"\nheld-out frame: {bad.mean()*100:.2f}% of pixels should be rejected")

    with torch.no_grad():
        xin = torch.from_numpy(((ev[..., :IN_CH] - mu) / sd)).permute(2, 0, 1)[None]
        pred = model(xin)[0, 0].numpy()
    clas = classical_R(ev)

    def score(R, name):
        det = (R[bad] < 0.5).mean() * 100 if bad.any() else float("nan")
        fp = (R[~bad] < 0.5).mean() * 100
        print(f"  {name:12s} catches {det:5.1f}% of harmful pixels, "
              f"falsely rejects {fp:5.1f}% of good ones  (mean R {R.mean():.3f})")
    score(clas, "classical")
    score(pred, "learned")

    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd},
               os.path.join(SC, "robnet.pt"))
    with open(os.path.join(SC, "robnet_norm.json"), "w") as f:
        json.dump({"mu": mu.tolist(), "sd": sd.tolist(), "in_ch": IN_CH}, f)
    print("\nsaved robnet.pt")
