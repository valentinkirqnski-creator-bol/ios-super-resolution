"""Train sr_gate against the MERGED image.

    python pack.py  --data data
    python train.py --data data --minutes 28 --out sr_gate.pt

The loss is the only interesting thing here. It is not a distance to Wronski's
R, and it is not a distance to any per-pixel "correct" R, because no such label
exists: the merge is a super-resolution merge, and a sub-pixel offset between
frames is the signal it feeds on rather than damage. A label of the form
1/(1 + delta^2/sigma^2) calls every offset damage, and the project has already
measured that a network trained faithfully to such a label is NET HARMFUL below
about 1.6 px of flow error for exactly that reason.

So the target is the merged image the pipeline would have produced with perfect
alignment and nothing rejected:

    loss = || merge(A, B, gate(features)) - merge(true flow, no noise, R = 1) ||

Both sides go through the same merge, the same kernels, the same CFA and the
same bounds handling, and R enters that merge exactly as it does in
core/merge.cpp (tools/sr_gate/probe_oracle.py checks the two agree to 4e-8).
Aliased, well-aligned regions therefore pay for being rejected -- the target has
the detail their offsets contribute -- and misaligned regions pay for being
merged, because the target does not have their ghost. Nothing has to be
hand-weighted to balance those two; the merge prices both.

Two terms are added to the plain L1:

  * edge weighting, because ghosting and lost detail both live on edges and a
    flat mean over a mostly flat photo hides them;
  * a gradient-matching term on the luminance, which is what actually reads as
    doubling or smearing to a viewer and which a pixel L1 is largely blind to.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate

FIELDS = ('feat', 'A', 'B', 'A_ref', 'B_ref', 'gt', 'edge', 'Rw', 'img',
          'ferr', 'art', 'meta')


class Split:
    """Memory-mapped burst stack; batches are materialised as fp32 on demand."""

    def __init__(self, root, name):
        self.d = {}
        for k in FIELDS:
            p = os.path.join(root, '%s_%s.npy' % (name, k))
            self.d[k] = np.load(p, mmap_mode='r')
        self.n = self.d['feat'].shape[0]

    def batch(self, idx, keys=FIELDS):
        out = {}
        for k in keys:
            if k == 'meta':
                out[k] = np.asarray(self.d[k][idx])
            else:
                out[k] = torch.from_numpy(
                    np.ascontiguousarray(self.d[k][idx], dtype=np.float32))
        return out


def lum_grad(x):
    """Gradient of the luminance, the channel a viewer reads doubling in."""
    g = x.mean(dim=-3, keepdim=True)
    return g[..., :, 1:] - g[..., :, :-1], g[..., 1:, :] - g[..., :-1, :]


def losses(out, gt, edge, w_edge=3.0, w_grad=1.0):
    l1 = (out - gt).abs().mean()
    norm = edge.flatten(1).mean(dim=1).clamp_min(1e-6).view(-1, 1, 1)
    ew = 1.0 + w_edge * (edge / norm).clamp(0, 8)
    l1e = ((out - gt).abs().mean(dim=1) * ew).mean()
    ax, ay = lum_grad(out)
    bx, by = lum_grad(gt)
    lg = (ax - bx).abs().mean() + (ay - by).abs().mean()
    return l1, l1e, lg, l1e + w_grad * lg


def psnr_t(out, gt):
    m = ((out - gt) ** 2).mean().item()
    return 10.0 * np.log10(1.0 / max(m, 1e-12))


def run_gate(net, feat):
    """feat (batch, N, C, h, w) -> R (batch, N, h, w).

    Sliced to the channel count the NETWORK declares, not the dataset's. The
    feature set only ever grows by appending, so an 8-channel checkpoint scores
    correctly on a 12-channel dataset -- which is what makes an added-features
    A/B exact: both models see the same bursts and the same noise draws, rather
    than two datasets built from separate random seeds.
    """
    want = net.convs[0].weight.shape[1]
    if feat.shape[2] > want:
        feat = feat[:, :, :want]
    b, n = feat.shape[0], feat.shape[1]
    r = net(feat.flatten(0, 1))[:, 0]
    return r.view(b, n, feat.shape[-2], feat.shape[-1])


def evaluate(net, val, base=None):
    """PSNR of Wronski / R=1 / the gate over the validation grid.

    With a base net given, also the base's PSNR and mask mean, which is the
    comparison that matters for a correction run: not "is this better than
    Wronski" but "does this improve on what already ships"."""
    net.eval()
    rows = []
    with torch.no_grad():
        for i in range(val.n):
            d = val.batch([i])
            args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
            Rg = run_gate(net, d['feat'])
            pg = psnr_t(gate.merge(*args, Rg), d['gt'])
            pw = psnr_t(gate.merge(*args, d['Rw']), d['gt'])
            p1 = psnr_t(gate.merge(*args, torch.ones_like(d['Rw'])), d['gt'])
            mr = float(Rg.mean())
            if base is None:
                pb, mb = float('nan'), float('nan')
            else:
                Rb = run_gate(base, d['feat'])
                pb = psnr_t(gate.merge(*args, Rb), d['gt'])
                mb = float(Rb.mean())
            rows.append((int(d['meta'][0][0]), float(d['meta'][0][1]),
                         pw, p1, pg, mr, float(d['Rw'].mean()), pb, mb))
    net.train()
    return rows


def report(rows):
    names = {0: 'jitter', 1: 'rot-only', 2: 'object', 3: 'outlier',
             4: 'parallax'}
    has_base = not np.isnan(rows[0][7])
    if has_base:
        print('%-9s %-6s %-8s %-8s %-8s %-8s %-9s %-9s %-7s %-7s' %
              ('regime', 'sig_f', 'Wronski', 'R=1', 'base', 'gate',
               'vs base', 'vs R=1', 'meanRg', 'meanRb'))
        for reg, sig, pw, p1, pg, mr, mw, pb, mb in sorted(
                rows, key=lambda r: (r[0], r[1])):
            print('%-9s %-6.2f %-8.2f %-8.2f %-8.2f %-8.2f %+-9.2f %+-9.2f '
                  '%-7.3f %-7.3f'
                  % (names[reg], sig, pw, p1, pb, pg, pg - pb, pg - p1, mr, mb))
        a = np.array([[r[2], r[3], r[4], r[7], r[5], r[8]] for r in rows])
        print('%-9s %-6s %-8.2f %-8.2f %-8.2f %-8.2f %+-9.2f %+-9.2f %-7.3f '
              '%-7.3f'
              % ('MEAN', '', a[:, 0].mean(), a[:, 1].mean(), a[:, 3].mean(),
                 a[:, 2].mean(), (a[:, 2] - a[:, 3]).mean(),
                 (a[:, 2] - a[:, 1]).mean(), a[:, 4].mean(), a[:, 5].mean()))
        # Selected on gain over the BASE. Gain over Wronski would be dominated
        # by what the base already won and would barely move between steps.
        return float((a[:, 2] - a[:, 3]).mean())
    print('%-9s %-6s %-8s %-8s %-8s %-10s %-9s %-7s %-7s' %
          ('regime', 'sig_f', 'Wronski', 'R=1', 'gate', 'vs Wronski', 'vs R=1',
           'meanRg', 'meanRw'))
    for reg, sig, pw, p1, pg, mr, mw, _pb, _mb in sorted(
            rows, key=lambda r: (r[0], r[1])):
        print('%-9s %-6.2f %-8.2f %-8.2f %-8.2f %+-10.2f %+-9.2f %-7.3f %-7.3f'
              % (names[reg], sig, pw, p1, pg, pg - pw, pg - p1, mr, mw))
    a = np.array([[r[2], r[3], r[4]] for r in rows])
    print('%-9s %-6s %-8.2f %-8.2f %-8.2f %+-10.2f %+-9.2f'
          % ('MEAN', '', a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(),
             (a[:, 2] - a[:, 0]).mean(), (a[:, 2] - a[:, 1]).mean()))
    return float((a[:, 2] - a[:, 0]).mean())


def save_ck(net, **extra):
    if isinstance(net, gate.SRGateOnlyStricter):
        return net.checkpoint(**extra)
    return gate.plain_checkpoint(net, **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--out', default='sr_gate.pt')
    ap.add_argument('--minutes', type=float, default=28.0)
    ap.add_argument('--steps', type=int, default=0,
                    help='fixed step count instead of a time budget. Comparing two '
                         'feature sets under the same WALL CLOCK is not a fair '
                         'test: a wider input is slower per step, so it silently '
                         'gets fewer of them. Match steps when comparing.')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=4e-3)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-coarse', action='store_true',
                    help='fine branch only, the 13x13 receptive field that could '
                         'not see a whole moving subject')
    ap.add_argument('--in-ch', type=int, default=0,
                    help='train on only the first N feature channels. The set only '
                         'ever grows by appending, so this ablates the additions '
                         'against the SAME bursts rather than a rebuilt dataset.')
    ap.add_argument('--base', default='',
                    help='checkpoint to freeze as R_base and train a correction '
                         'on top of, as R = R_base * sigmoid(corr). The result '
                         'is then <= R_base at every pixel BY CONSTRUCTION, so '
                         'it cannot come out more permissive than what ships, '
                         'whatever the loss prefers.')
    ap.add_argument('--eval-every', type=float, default=300.0,
                    help='seconds between validation passes')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    train = Split(root, 'train')
    val = Split(root, 'val')
    print('bursts: %d train, %d val  (memory mapped)' % (train.n, val.n))

    base_net = None
    if a.base:
        bp = a.base if os.path.isabs(a.base) else os.path.join(here, a.base)
        base_net = gate.from_checkpoint(torch.load(bp, map_location='cpu',
                                                   weights_only=False))
        bch = base_net.convs[0].weight.shape[1]
        corr = gate.SRGate(in_ch=(base_net.cconvs[0].weight.shape[1] // 2)
                           if base_net.coarse else bch,
                           dilations=base_net.dilations, coarse=base_net.coarse,
                           pool=base_net.pool,
                           coarse_dilations=base_net.coarse_dilations)
        # The correction head starts at bias +2, so sigmoid(2) = 0.88: it opens
        # NEAR pass-through but not at it, because a saturated sigmoid has no
        # gradient and the run would never start moving.
        net = gate.SRGateOnlyStricter(base_net, corr)
        print('only-stricter: %d trainable parameters on a frozen base from %s'
              % (net.n_params(), a.base))
    else:
        net = gate.SRGate(in_ch=a.in_ch or gate.NUM_FEATURES,
                          coarse=not a.no_coarse)
        print('sr_gate: %d parameters, %d input channels'
              % (net.n_params(), net.convs[0].weight.shape[1]))
    opt = torch.optim.Adam([q for q in net.parameters() if q.requires_grad],
                           lr=a.lr)
    budget = a.minutes * 60.0
    t0 = time.time()

    # Estimate the step count from the first few steps so the one-cycle
    # schedule actually completes inside the time budget instead of being cut
    # off mid-ramp -- a cut-off cosine leaves the weights at a high learning
    # rate, which is the difference between a usable and an unusable model.
    warm = 8
    for _ in range(warm):
        idx = sorted(rng.choice(train.n, size=a.batch, replace=False).tolist())
        d = train.batch(idx)
        out = gate.merge(d['A_ref'], d['B_ref'], d['A'], d['B'],
                         run_gate(net, d['feat']))
        _, _, _, loss = losses(out, d['gt'], d['edge'])
        opt.zero_grad()
        loss.backward()
        opt.step()
    per_step = (time.time() - t0) / warm
    if a.steps > 0:
        total_steps = a.steps
        budget = 1e9        # the step count is the budget now
        print('%.3f s/step -> %d steps requested (~%.0f min)'
              % (per_step, total_steps, per_step * total_steps / 60.0))
    else:
        total_steps = max(200, int((budget - (time.time() - t0)) / per_step))
        print('%.3f s/step -> planning %d steps' % (per_step, total_steps))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=total_steps, pct_start=0.15)

    best = -1e9
    outp = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)
    log_t = time.time()
    eval_t = time.time()
    step = 0
    while step < total_steps and time.time() - t0 < budget:
        idx = sorted(rng.choice(train.n, size=a.batch, replace=False).tolist())
        d = train.batch(idx, ('feat', 'A', 'B', 'A_ref', 'B_ref', 'gt', 'edge'))
        R = run_gate(net, d['feat'])
        out = gate.merge(d['A_ref'], d['B_ref'], d['A'], d['B'], R)
        l1, l1e, lg, loss = losses(out, d['gt'], d['edge'])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1

        if time.time() - log_t > 45.0:
            log_t = time.time()
            print('  step %5d/%d  %4.0fs  l1 %.5f  edge %.5f  grad %.5f  '
                  'meanR %.3f  lr %.1e'
                  % (step, total_steps, time.time() - t0, l1.item(), l1e.item(),
                     lg.item(), R.mean().item(), sched.get_last_lr()[0]),
                  flush=True)
        if time.time() - eval_t > a.eval_every:
            eval_t = time.time()
            g = report(evaluate(net, val, base_net))
            print('  [eval @ step %d] mean gain vs %s %+.2f dB'
                  % (step, 'base' if base_net else 'Wronski', g), flush=True)
            if g > best:
                best = g
                torch.save(save_ck(net, steps=step, val_gain=g), outp)
                print('  saved (best so far)', flush=True)

    print('trained %d steps in %.0fs' % (step, time.time() - t0))
    g = report(evaluate(net, val, base_net))
    print('final mean gain vs %s %+.2f dB (best checkpoint %+.2f)'
          % ('base' if base_net else 'Wronski', g, best))
    if g >= best:
        torch.save(save_ck(net, steps=step, val_gain=g), outp)
        print('saved final to', outp)
    else:
        print('kept the earlier checkpoint at', outp)


if __name__ == '__main__':
    main()
