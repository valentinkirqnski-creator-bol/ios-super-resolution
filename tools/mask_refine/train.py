"""Train the mask refinement against the FINAL MERGE.

    python gen_bursts.exe APC_1186.dng data 60 5 512
    python train.py --data data --minutes 25

Selected on reconstruction over held-out bursts, with the retained-sample ratio
reported beside it: a mask that wins on PSNR by rejecting most of the burst has
not won anything, and the pair makes that visible instead of hiding it in one
number.

The time budget is a hard wall-clock limit, and the one-cycle schedule is sized
from a measured step time so it completes inside the budget rather than being
cut off at a high learning rate.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import losses as L
import model as M


def psnr(a, b):
    m = float(((a - b) ** 2).mean())
    return 10.0 * np.log10(1.0 / max(m, 1e-12))


def evaluate(net, bursts, idx):
    net.eval()
    rows = []
    with torch.no_grad():
        for i in idx:
            feat, Rw, A_ref, B_ref, A, B, gt, ferr = bursts.batch(i, window=0)
            Mf, _ = net(feat, Rw)
            out = D.merge(A_ref, B_ref, A, B, Mf)
            ow = D.merge(A_ref, B_ref, A, B, Rw)
            rows.append((psnr(out, gt), psnr(ow, gt),
                         float(Mf.sum() / Rw.sum().clamp_min(1.0)),
                         float(Rw.mean()), float(Mf.mean())))
    net.train()
    v = np.array(rows)
    return dict(psnr=v[:, 0].mean(), psnr_w=v[:, 1].mean(),
                retained=v[:, 2].mean(), meanW=v[:, 3].mean(),
                meanF=v[:, 4].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--out', default='mask_refine.pt')
    ap.add_argument('--minutes', type=float, default=25.0)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--width', type=int, default=16)
    ap.add_argument('--head-bias', type=float, default=1.0)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--window', type=int, default=256,
                    help='output-pixel window sampled per step. The merge is '
                         'pointwise given A and B, so a window is exactly what '
                         'the pipeline produces there: it buys many more steps '
                         'inside the budget and is free augmentation.')
    ap.add_argument('--cache', type=int, default=6)
    ap.add_argument('--w-recon', type=float, default=1.0)
    ap.add_argument('--w-ghost', type=float, default=1.0)
    ap.add_argument('--w-suppress', type=float, default=0.10)
    ap.add_argument('--w-distill', type=float, default=0.0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    t_load = time.time()
    bursts = D.Bursts(root, cache=a.cache, window=a.window, seed=a.seed)
    n = len(bursts)
    n_val = max(1, int(round(n * a.val_frac)))
    perm = rng.permutation(n)
    val_idx, train_idx = list(perm[:n_val]), list(perm[n_val:])
    print('%d bursts (%d train, %d held out) loaded in %.0fs'
          % (n, len(train_idx), len(val_idx), time.time() - t_load), flush=True)

    net = M.MaskRefineNet(in_ch=D.NUM_FEATURES, width=a.width,
                          head_bias=a.head_bias)
    print('model: %d parameters, receptive field %d mask px, features %s'
          % (net.n_params(), net.receptive_field(), ', '.join(D.FEATURES)),
          flush=True)
    w = dict(recon=a.w_recon, ghost=a.w_ghost, suppress=a.w_suppress,
             distill=a.w_distill)
    print('loss weights %s' % w, flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    budget = a.minutes * 60.0
    t0 = time.time()

    # Measure a step, then size the schedule to the remaining budget.
    for i in range(4):
        feat, Rw, A_ref, B_ref, A, B, gt, ferr = bursts.batch(
            train_idx[i % len(train_idx)])
        Mf, _ = net(feat, Rw)
        with torch.no_grad():
            ow = D.merge(A_ref, B_ref, A, B, Rw)
        loss, _ = L.total(D.merge(A_ref, B_ref, A, B, Mf), ow, gt, Rw, Mf, ferr, w)
        opt.zero_grad(); loss.backward(); opt.step()
    per_step = (time.time() - t0) / 4.0
    steps = max(50, int((budget - (time.time() - t0)) / per_step))
    print('%.2f s/step -> %d steps in the %.0f min budget'
          % (per_step, steps, a.minutes), flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=steps,
                                                pct_start=0.2)

    best, log_t, outp = -1e9, time.time(), os.path.join(here, a.out)
    base = evaluate(net, bursts, val_idx)
    print('before training: PSNR %.2f (Wronski %.2f)  retained %.3f'
          % (base['psnr'], base['psnr_w'], base['retained']), flush=True)

    step = 0
    while step < steps and time.time() - t0 < budget:
        i = train_idx[int(rng.integers(len(train_idx)))]
        feat, Rw, A_ref, B_ref, A, B, gt, ferr = bursts.batch(i)
        Mf, _ = net(feat, Rw)
        out = D.merge(A_ref, B_ref, A, B, Mf)
        with torch.no_grad():
            out_w = D.merge(A_ref, B_ref, A, B, Rw)
        loss, parts = L.total(out, out_w, gt, Rw, Mf, ferr, w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1

        if time.time() - log_t > 30.0:
            log_t = time.time()
            e = evaluate(net, bursts, val_idx)
            print('  step %5d/%d %4.0fs  recon %.4f ghost %.4f supp %.3f | '
                  'val PSNR %.2f vs %.2f  retained %.3f  lr %.1e'
                  % (step, steps, time.time() - t0, parts['recon'],
                     parts['ghost'], parts['suppress'], e['psnr'],
                     e['psnr_w'], e['retained'], sched.get_last_lr()[0]),
                  flush=True)
            # Selected on PSNR gain over the Wronski mask, with a floor on how
            # much of the burst may be thrown away to get it.
            score = (e['psnr'] - e['psnr_w']) - 2.0 * max(0.0, 0.5 - e['retained'])
            if score > best:
                best = score
                torch.save(dict(state_dict=net.state_dict(), width=a.width,
                                in_ch=D.NUM_FEATURES, features=D.FEATURES,
                                step=step, score=score, val=e), outp)
                print('    saved (best)', flush=True)

    e = evaluate(net, bursts, val_idx)
    print('done: %d steps in %.1f min' % (step, (time.time() - t0) / 60.0))
    print('final  PSNR %.2f  Wronski %.2f  gain %+.2f dB  retained %.3f  '
          'meanM %.3f -> %.3f'
          % (e['psnr'], e['psnr_w'], e['psnr'] - e['psnr_w'], e['retained'],
             e['meanW'], e['meanF']))
    print('best checkpoint at', outp)


if __name__ == '__main__':
    main()
