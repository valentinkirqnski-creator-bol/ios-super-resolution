"""Do the added channels carry signal the original eight do not?

    python probe_features.py --data data --split test --bursts 8

Independent of training, which is the point: a training run that comes out ahead
by 0.2 dB could be seed noise, whereas this asks directly whether the information
is present. Two probes, both against the ORACLE mask -- R optimised per burst
straight against the ground truth, so it is the best any mask of this shape could
do (probe_oracle.py):

  1. rank correlation of each channel with the oracle mask, on its own;
  2. a ridge linear probe's R^2 on channels 0-7 versus 0-11, which is the
     INCREMENTAL value -- a channel can correlate well and still add nothing if
     the first eight already span it.

A linear probe understates what a 3-layer network extracts, so read it as a floor
rather than a prediction.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate
import srsim
import train as T


def oracle_mask(d, steps=250, lr=0.25):
    """R optimised per burst against the ground truth (has seen the answer)."""
    A_ref, B_ref, A, B, gt = (d['A_ref'], d['B_ref'], d['A'], d['B'], d['gt'])
    logit = torch.full_like(d['Rw'], 2.0).requires_grad_(True)
    opt = torch.optim.Adam([logit], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        out = gate.merge(A_ref, B_ref, A, B, torch.sigmoid(logit))
        (out - gt).abs().mean().backward()
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(logit)


def ridge_r2(X, y, lam=1e-3):
    """R^2 of a ridge fit, on held-out halves so extra columns cannot inflate it."""
    n = X.shape[0]
    half = n // 2
    Xa, ya, Xb, yb = X[:half], y[:half], X[half:], y[half:]
    out = []
    for (Xt, yt, Xv, yv) in ((Xa, ya, Xb, yb), (Xb, yb, Xa, ya)):
        Xt1 = np.hstack([Xt, np.ones((Xt.shape[0], 1))])
        Xv1 = np.hstack([Xv, np.ones((Xv.shape[0], 1))])
        G = Xt1.T @ Xt1 + lam * np.eye(Xt1.shape[1])
        w = np.linalg.solve(G, Xt1.T @ yt)
        pred = Xv1 @ w
        ss_res = float(((yv - pred) ** 2).sum())
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        out.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--split', default='test')
    ap.add_argument('--bursts', type=int, default=8)
    ap.add_argument('--stride', type=int, default=3,
                    help='subsample pixels; the mask is smooth, so every pixel is '
                         'not an independent sample and all of them is just slower')
    ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    sp = T.Split(root, a.split)

    Xs, ys = [], []
    n = min(a.bursts, sp.n)
    # Spread the sample across the grid rather than taking the first few, which
    # are all one regime.
    picks = np.linspace(0, sp.n - 1, n).astype(int)
    for i in picks:
        d = sp.batch([int(i)])
        Ro = oracle_mask(d)[0].numpy()          # (N, h, w)
        f = d['feat'][0].numpy()                # (N, C, h, w)
        s = a.stride
        # Channel axis FIRST before flattening. f is (N, C, h, w), so reshaping
        # straight to (C, -1) folds frames into channels and every column comes
        # out a blend of all twelve -- which shows up as every channel having the
        # same standard deviation and no correlation with anything.
        fc = f[:, :, ::s, ::s].transpose(1, 0, 2, 3)      # (C, N, h', w')
        Xs.append(fc.reshape(fc.shape[0], -1).T)
        ys.append(Ro[:, ::s, ::s].reshape(-1))
        print('  burst %2d  regime %d  sigma_f %.2f  oracle meanR %.3f'
              % (i, int(d['meta'][0][0]), float(d['meta'][0][1]), Ro.mean()),
              flush=True)
    X = np.concatenate(Xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    print('samples %d, channels %d' % (X.shape[0], X.shape[1]))

    print()
    print('%-10s %-12s %-12s' % ('channel', 'spearman', 'std'))
    order = np.argsort(y)
    ry = np.empty_like(y)
    ry[order] = np.arange(y.size)
    for c in range(X.shape[1]):
        o = np.argsort(X[:, c])
        rx = np.empty_like(y)
        rx[o] = np.arange(y.size)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        print('%-10s %+-12.3f %-12.4f' % (srsim.FEATURE_NAMES[c], rho, X[:, c].std()))

    print()
    r8 = ridge_r2(X[:, :8], y)
    r12 = ridge_r2(X, y)
    print('ridge R^2 on channels 0-7 : %.4f' % r8)
    print('ridge R^2 on channels 0-11: %.4f' % r12)
    print('incremental from 8-11     : %+.4f' % (r12 - r8))
    for c in range(8, X.shape[1]):
        cols = list(range(8)) + [c]
        print('   + %-9s alone      : %+.4f'
              % (srsim.FEATURE_NAMES[c], ridge_r2(X[:, cols], y) - r8))


if __name__ == '__main__':
    main()
