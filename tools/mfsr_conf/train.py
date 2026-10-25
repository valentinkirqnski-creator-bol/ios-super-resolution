"""Train ConfNet.

    python train.py configs/train_m1.yaml [--minutes 40]

Loss (all terms per raw pixel / output pixel, frames of a burst padded to K_MAX and masked):
  L_align  = BCE(A, t_align),  t_align = vis * exp(-e^2 / (2 sigma_a^2)),  weight 1 + w_misalign (1 - t_align)
             geometry only: does the sample come from the right scene point?
  L_merge  = BCE(R, R*),       R* = reconstruction-optimal mask (labels.py),  weight 1 + w_harm (1 - R*)
  L_recon  = the label's own reconstruction cost of the app's merge for the predicted R (merge.cost_map "linear",
             exact replica, noise-variance units): gamma (M - I)^2 / sigma^2 + sum_k (1 - r_k)(g_k + n_k + prior).
             Its minimum IS R*, so it only re-weights the merge errors by how much they cost in the image (a wrong
             R at an edge costs more than at a flat pixel), it cannot pull R towards rejecting valid samples.
  L = lambda_align L_align + lambda_merge L_merge + lambda_recon L_recon        (configs: loss.*)
Validation (every val_every steps, the val split = different source scenes): the same terms, the excess
reconstruction cost over R* for the net / the classical mask / R = 1, valid-sample retention and harmful-sample
rejection. Best checkpoint = lowest validation reconstruction cost. Everything is logged to <out>/log.jsonl.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                       # noqa: E402
import torch                             # noqa: E402
import torch.nn.functional as F          # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mfsr import config as C, merge as mg, model as M          # noqa: E402
from mfsr.data import Bursts, collate, parts_of, K_MAX           # noqa: E402


def _wi(_):
    torch.set_num_threads(1)


def forward(net, batch, halo, use_r0):
    """-> A, R [B,K,Pc+1,Pc+1] on the labelled raw pixels, and the logits."""
    r0 = batch["r0"] if use_r0 else None
    lg = net(batch["xg"], batch["xr"], batch["mask"], r0)
    Pc1 = batch["R"].shape[-1]
    lg = lg[..., halo:halo + Pc1, halo:halo + Pc1]
    return torch.sigmoid(lg[:, :, 0]), torch.sigmoid(lg[:, :, 1]), lg


def losses(A, R, lg, batch, Lc):
    m = batch["mask"][:, :, None, None]
    ta = batch["vis"] * torch.exp(-batch["e"] ** 2 / (2.0 * Lc["align_sigma"] ** 2))
    wa = m * (1.0 + Lc["w_misalign"] * (1.0 - ta))
    la = (wa * F.binary_cross_entropy_with_logits(lg[:, :, 0], ta, reduction="none")).sum() / wa.sum()
    Rs = batch["R"]
    wm = m * (1.0 + Lc["w_harm"] * (1.0 - Rs))
    lm = (wm * F.binary_cross_entropy_with_logits(lg[:, :, 1], Rs, reduction="none")).sum() / wm.sum()
    parts = parts_of(batch)
    Rp = R * m + (1.0 - m)                         # padded frames: weight irrelevant (no samples), keep at 1
    c = mg.cost_map(parts, parts.fetch(Rp), batch["T"], batch["sig2"], Lc["gamma"], Lc["keep_prior"], "linear")
    lr_ = c.mean()
    return la, lm, lr_, parts


@torch.no_grad()
def validate(net, dl, halo, use_r0, Lc):
    net.eval()
    acc = {}
    def add(k, v, n=1):
        s, c = acc.get(k, (0.0, 0))
        acc[k] = (s + float(v) * n, c + n)
    for batch in dl:
        A, R, lg = forward(net, batch, halo, use_r0)
        la, lm, lr_, parts = losses(A, R, lg, batch, Lc)
        m = batch["mask"][:, :, None, None]
        def cost(Rmap):
            Rp = Rmap * m + (1.0 - m)
            return mg.cost_map(parts, parts.fetch(Rp), batch["T"], batch["sig2"], Lc["gamma"], Lc["keep_prior"],
                               "linear").mean().item()
        r0 = batch["r0"][..., halo:halo + R.shape[-1], halo:halo + R.shape[-1]]
        c_star, c_net, c_r0, c_one = cost(batch["R"]), lr_.item(), cost(r0), cost(torch.ones_like(R))
        add("loss_align", la), add("loss_merge", lm), add("cost_net", c_net), add("cost_oracle", c_star)
        add("cost_r0", c_r0), add("cost_ones", c_one)
        valid = (batch["R"] >= 0.5) & (m > 0)
        harm = (batch["R"] < 0.5) & (m > 0)
        if valid.any():
            add("valid_kept", ((R >= 0.5) & valid).sum() / valid.sum(), int(valid.sum()))
            add("valid_kept_r0", ((r0 >= 0.5) & valid).sum() / valid.sum(), int(valid.sum()))
        if harm.any():
            add("harm_rejected", ((R < 0.5) & harm).sum() / harm.sum(), int(harm.sum()))
            add("harm_rejected_r0", ((r0 < 0.5) & harm).sum() / harm.sum(), int(harm.sum()))
    net.train()
    out = {k: s / max(c, 1) for k, (s, c) in acc.items()}
    for k in ("net", "r0", "ones"):
        out[f"excess_{k}"] = out[f"cost_{k}"] - out["cost_oracle"]
    return out


def input_stats(ds, n=24):
    """Per-channel mean / std of the inputs over real frames of a few training items."""
    sg, sg2, sr, sr2, cg, cr = 0, 0, 0, 0, 0, 0
    for i in np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int):
        for c in ds[int(i)]:
            K = c["K"]
            xg, xr = c["xg"][:K], c["xr"][:K]
            sg = sg + xg.sum((0, 2, 3)); sg2 = sg2 + (xg ** 2).sum((0, 2, 3)); cg += xg[:, 0].numel()
            sr = sr + xr.sum((0, 2, 3)); sr2 = sr2 + (xr ** 2).sum((0, 2, 3)); cr += xr[:, 0].numel()
    mg_, mr_ = sg / cg, sr / cr
    return mg_, (sg2 / cg - mg_ ** 2).clamp_min(1e-6).sqrt(), mr_, (sr2 / cr - mr_ ** 2).clamp_min(1e-6).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--train_split", default="train", help="debug only: train on another split")
    ap.add_argument("--val_limit", type=int, default=0)
    a = ap.parse_args()
    cfg = C.load_train(a.config)
    if a.minutes is not None:
        cfg["minutes"] = a.minutes
    out = a.out or os.path.join(HERE, "runs", cfg["name"])
    os.makedirs(out, exist_ok=True)
    C.save(cfg, os.path.join(out, "train_config.yaml"))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    root = os.path.join(HERE, cfg["data"]) if not os.path.isabs(cfg["data"]) else cfg["data"]
    mc, Lc = cfg["model"], cfg["loss"]
    use_r0 = bool(mc.get("use_r0") or mc.get("refine_r0"))
    tr = Bursts(root, a.train_split, crop=cfg["crop"], halo=cfg["halo"], n_crops=cfg.get("n_crops", 2), random_crop=True,
                use_r0=use_r0)
    va = Bursts(root, "val", crop=cfg.get("val_crop", 128), halo=cfg["halo"], random_crop=False, use_r0=use_r0,
                limit=a.val_limit or cfg.get("val_limit", 0))
    print(f"train items {len(tr)}, val items {len(va)}", flush=True)
    net = M.build(mc)
    mu_g, sd_g, mu_r, sd_r = input_stats(tr)
    net.mu_g.copy_(mu_g); net.sd_g.copy_(sd_g); net.mu_r.copy_(mu_r); net.sd_r.copy_(sd_r)
    print(f"params {net.n_params()}, MACs per quad per frame {net.macs_per_quad()}", flush=True)
    dl = DataLoader(tr, batch_size=cfg["batch"], shuffle=True, num_workers=cfg["workers"], collate_fn=collate,
                    worker_init_fn=_wi, persistent_workers=cfg["workers"] > 0, drop_last=True)
    dv = DataLoader(va, batch_size=2, shuffle=False, num_workers=cfg["workers"], collate_fn=collate,
                    worker_init_fn=_wi)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    log = open(os.path.join(out, "log.jsonl"), "a")
    # The budget counts ACTIVE time: a laptop that sleeps mid-run (measured: 37 min lost on the first m1 run) pauses
    # training instead of eating the budget. Any gap above 120 s between steps counts as 120 s.
    t0, step, best = time.time(), 0, None
    t_active, t_last = 0.0, time.time()
    budget = cfg["minutes"] * 60.0
    run = None
    v0 = validate(net, dv, cfg["halo"], use_r0, Lc)
    v0.update(step=0, t=0.0)
    log.write(json.dumps(v0) + "\n"); log.flush()
    print("val@0", {k: round(v, 4) for k, v in v0.items()}, flush=True)
    done = False
    while not done:
        for batch in dl:
            now = time.time()
            t_active += min(now - t_last, 120.0)
            t_last = now
            frac = min(1.0, t_active / budget)
            for gq in opt.param_groups:
                gq["lr"] = cfg["lr"] * 0.5 * (1 + math.cos(math.pi * frac))
            A, R, lg = forward(net, batch, cfg["halo"], use_r0)
            la, lm, lr_, _ = losses(A, R, lg, batch, Lc)
            loss = Lc["align"] * la + Lc["merge"] * lm + Lc["recon"] * lr_
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            step += 1
            cur = dict(la=la.item(), lm=lm.item(), lr=lr_.item())
            run = cur if run is None else {k: 0.95 * run[k] + 0.05 * cur[k] for k in cur}
            if step % 25 == 0:
                print(f"step {step} {t_active:.0f}s active " + " ".join(f"{k} {v:.4f}" for k, v in run.items()),
                      flush=True)
            end = t_active >= budget or step >= cfg["max_steps"]
            if step % cfg["val_every"] == 0 or end:
                v = validate(net, dv, cfg["halo"], use_r0, Lc)
                v.update(step=step, t=t_active, train=run)
                t_last = time.time()                     # validation time is not training time
                log.write(json.dumps(v) + "\n"); log.flush()
                print(f"val@{step}", {k: round(x, 4) for k, x in v.items() if not isinstance(x, dict)}, flush=True)
                ck = dict(state=net.state_dict(), cfg=cfg, step=step, val=v)
                torch.save(ck, os.path.join(out, "last.pt"))
                if best is None or v["cost_net"] < best:
                    best = v["cost_net"]
                    torch.save(ck, os.path.join(out, "best.pt"))
            if end:
                done = True
                break
    print(f"done: {step} steps in {t_active:.0f}s active ({time.time() - t0:.0f}s wall), best val cost {best:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
