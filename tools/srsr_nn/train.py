"""Train / evaluate the learned SRSR robustness mask end to end, through the exact
SRSR merge (srsr_merge.py), against the noise-free target.

Loss (per output pixel, in units of the pixel's noise std, so dark and bright
areas count alike):
    charbonnier(out - target) + w_grad * charbonnier(grad(out) - grad(target))
weighted by 1 + gamma * ghost, ghost = how much keeping the worst comparison frame
would make the pixel worse than the reference alone (data.ghost_map). Without the
weight an image loss prefers keeping a ghost (a few pixels) to losing noise
reduction (every pixel); the weight makes ghosts expensive.

Training never starts without --go (the user starts training).

    python train.py --data D:/srsr_nn_data/train --val D:/srsr_nn_data/val --model big --out runs/big1 --go
    python train.py --val D:/srsr_nn_data/val --init runs/big1/model.pt --eval
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import data as D
import model as M
import srsr_merge as sm


def charb(x, eps=1e-2):
    return torch.sqrt(x * x + eps * eps) - eps


def run_window(net, w, masks_override=None):
    """Masks for every comparison frame of a window, the merged window, and its loss terms."""
    F_ = w["mov"].shape[0]
    if masks_override is None:
        ref = w["ref"][None].expand(F_, -1, -1, -1)
        masks = net(ref, w["mov"], w["pos"], w["flow"], w["analytic"],
                    w["a"][0:1].expand(F_, -1), w["b"][0:1].expand(F_, -1), w["a"][1:], w["b"][1:])[:, 0]
    else:
        masks = masks_override
    out = sm.merge_uv(w["terms"], masks, w["x0"], w["y0"], w["uv"])
    return masks, out


def loss_of(out, w, gamma, w_grad):
    h, wd = w["out_hw"]
    d = (out - w["target"]) / w["sigma"]
    g = w.get("_ghost")
    if g is None:
        g = w["_ghost"] = D.ghost_map(w)
    wt = 1.0 + gamma * ((g - 1.0).clamp(0, 3) / 3.0)                  # [P]
    l_img = (wt[:, None] * charb(d)).sum() / (3 * wt.sum())
    d2 = d.view(h, wd, 3)
    w2 = wt.view(h, wd, 1)
    gx = charb(d2[:, 1:] - d2[:, :-1]) * torch.maximum(w2[:, 1:], w2[:, :-1])
    gy = charb(d2[1:] - d2[:-1]) * torch.maximum(w2[1:], w2[:-1])
    l_grad = (gx.sum() + gy.sum()) / (3 * (w2[:, 1:].sum() + w2[1:].sum()))
    return l_img + w_grad * l_grad, g


@torch.no_grad()
def evaluate(net, files, args, max_windows=None):
    """Per-pixel noise-normalised error of: the network, the analytic mask, all frames kept,
    the reference only. Split into ghost pixels (keeping some frame hurts > 2 sigma) and the rest,
    plus the mean keep weight on each."""
    rng = np.random.default_rng(123)
    pool = D.Pool(files, 4)
    acc = {k: [0.0, 0.0, 0] for k in ("net", "analytic", "all", "ref")}
    keep = {"net": [0.0, 0.0], "analytic": [0.0, 0.0]}
    npx = [0, 0]
    nwin = 0
    for p in files:
        it = pool.get(p)
        for _ in range(args.eval_windows):
            w = D.window(it, args.window, args.halo, args.margin, rng)
            g = D.ghost_map(w)
            ghost = g > 2.0
            F_ = w["mov"].shape[0]
            variants = {
                "net": None if net is not None else w["analytic"][:, 0],
                "analytic": w["analytic"][:, 0],
                "all": torch.ones_like(w["analytic"][:, 0]),
                "ref": torch.zeros_like(w["analytic"][:, 0]),
            }
            for k, mo in variants.items():
                masks, out = run_window(net, w, mo)
                e = ((out - w["target"]) / w["sigma"]).abs().mean(-1)
                acc[k][0] += float(e[ghost].sum()); acc[k][1] += float(e[~ghost].sum())
                if k in keep:
                    r = sm.sample_crop(masks, w["x0"], w["y0"], w["uv"], w["terms"]["u"]["quadW"], w["terms"]["u"]["quadH"])
                    keep[k][0] += float(r[:, ghost].mean(0).sum()) if ghost.any() else 0.0
                    keep[k][1] += float(r[:, ~ghost].mean(0).sum())
            npx[0] += int(ghost.sum()); npx[1] += int((~ghost).sum())
            nwin += 1
            if max_windows and nwin >= max_windows:
                break
        if max_windows and nwin >= max_windows:
            break
    res = {k: dict(ghost=v[0] / max(1, npx[0]), clean=v[1] / max(1, npx[1])) for k, v in acc.items()}
    for k, v in keep.items():
        res[k].update(keep_ghost=v[0] / max(1, npx[0]), keep_clean=v[1] / max(1, npx[1]))
    res["ghost_px_frac"] = npx[0] / max(1, sum(npx))
    return res


def fmt(res):
    s = "ghost px %.2f%% | err(sigma) ghost/clean:" % (100 * res["ghost_px_frac"])
    for k in ("net", "analytic", "all", "ref"):
        s += "  %s %.3f/%.3f" % (k, res[k]["ghost"], res[k]["clean"])
    s += " | keep ghost/clean: net %.3f/%.3f analytic %.3f/%.3f" % (
        res["net"]["keep_ghost"], res["net"]["keep_clean"], res["analytic"]["keep_ghost"], res["analytic"]["keep_clean"])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--val"); ap.add_argument("--out", default="runs/run")
    ap.add_argument("--model", default="big", choices=list(M.CONFIGS))
    ap.add_argument("--init")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4, help="windows per step")
    ap.add_argument("--window", type=int, default=96, help="quads per side incl. halo")
    ap.add_argument("--halo", type=int, default=0, help="0 = the model's receptive field")
    ap.add_argument("--margin", type=int, default=12, help="comparison crop margin (quads)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=8.0, help="extra weight on ghost pixels")
    ap.add_argument("--w_grad", type=float, default=0.5)
    ap.add_argument("--eval_windows", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one forward/backward pass, no optimiser step, nothing saved")
    ap.add_argument("--go", action="store_true", help="required to train")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.halo == 0:
        args.halo = M.receptive_halo(args.model)
    net = M.build(args.model)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location="cpu"))
    nparam = sum(p.numel() for p in net.parameters())
    val = D.split(args.val)[0] if args.val else []

    if args.eval:
        net.eval()
        print(fmt(evaluate(net, val, args)))
        return
    files = D.split(args.data)[0]
    if args.smoke:
        rng = np.random.default_rng(0)
        w = D.window(D.Item(files[0]), args.window, args.halo, args.margin, rng)
        t0 = time.time()
        masks, out = run_window(net, w)
        loss, g = loss_of(out, w, args.gamma, args.w_grad)
        loss.backward()
        gn = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
        print("smoke: %s %d params, window %dx%d out px, %d frames, loss %.4f, grad norm %.3e, ghost>2 %.2f%%, %.2fs" % (
            args.model, nparam, *w["out_hw"], masks.shape[0], float(loss.detach()), gn, 100 * float((g > 2).float().mean()),
            time.time() - t0))
        return
    if not args.go:
        sys.exit("training not started: pass --go")

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    pool = D.Pool(files, 12)
    rng = np.random.default_rng(1)
    log = open(os.path.join(args.out, "log.txt"), "a")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train()
        opt.zero_grad()
        tot = 0.0
        for _ in range(args.batch):
            w = D.window(pool.get(files[int(rng.integers(len(files)))]), args.window, args.halo, args.margin, rng)
            _, out = run_window(net, w)
            loss, _ = loss_of(out, w, args.gamma, args.w_grad)
            (loss / args.batch).backward()
            tot += float(loss) / args.batch
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            msg = "step %d loss %.4f lr %.2e %.1f min" % (step, tot, sched.get_last_lr()[0], (time.time() - t0) / 60)
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if step % args.eval_every == 0 or step == args.steps:
            torch.save(net.state_dict(), os.path.join(args.out, "model_s%d.pt" % step))
            torch.save(net.state_dict(), os.path.join(args.out, "model.pt"))
            if val:
                net.eval()
                msg = "eval s%d: %s" % (step, fmt(evaluate(net, val, args)))
                print(msg, flush=True); log.write(msg + "\n"); log.flush()


if __name__ == "__main__":
    main()
