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

Three terms are added to the plain L1:

  * edge weighting, because ghosting and lost detail both live on edges and a
    flat mean over a mostly flat photo hides them;
  * a gradient-matching term on the luminance, which is what actually reads as
    doubling or smearing to a viewer and which a pixel L1 is largely blind to;
  * a GHOST term: extra weight where an edge and a real misalignment coincide
    (see ghost_risk). Edge weighting on its own was measured to produce a net
    that is uniformly cautious rather than selective -- mean R near 0.5
    everywhere, and only 0.231 on genuinely misaligned edges where Wronski
    reaches 0.021, which is visible as edge thickening even while the mean PSNR
    improves. The product of the two is what prices that specific failure.

The ghost term reads the true flow error, so a word on why that is not the R*
label this file opens by rejecting. It is used as a WEIGHT, never as a target:
it says a mistake at this pixel is expensive, not that the answer at this pixel
is zero. The target is still the merged image under perfect alignment, so an
aliased well-aligned edge is still paid for if it is rejected -- the weight just
stops a wrongly placed frame from being free.
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

FIELDS = ('feat', 'A', 'B', 'A_ref', 'B_ref', 'gt', 'edge', 'Rw', 'ferr',
          'meta')


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


def ghost_risk(ferr, hs, ws, thresh=1.6):
    """Where a REAL misalignment is available to ghost, on the loss grid.

    ferr (batch, N, gh, gw) is the exact error of the flow the merge used, in
    raw pixels. Reduced over frames with a max, because one badly placed frame
    is enough to thicken an edge -- an average would let seven good frames hide
    it. Then nearest-upsampled to the output grid the loss lives on.

    1.6 px is where this project measured rejection to stop being net harmful
    on the merged image. Below it an offset is mostly the sub-pixel signal the
    SR merge feeds on, so weighting those pixels harder would be asking the net
    to reject the thing it exists to preserve.

    Used ONLY as a loss weight. As a TARGET this quantity is the R* label that
    was measured to over-reject across the whole sub-pixel band; weighting says
    "a mistake here is expensive", which is a different statement from "the
    answer here is 0".
    """
    g = (ferr.amax(dim=1) > thresh).float().unsqueeze(1)
    return torch.nn.functional.interpolate(g, size=(hs, ws), mode='nearest')[:, 0]


def keep_value(ferr, hs, ws, thresh=1.6):
    """Where EVERY frame is well aligned, so merging is unambiguously right.

    The exact complement of ghost_risk: amax <= thresh against its amax >
    thresh. A min over frames was tried first and is wrong -- "at least one of
    seven frames is usable" is true for 99.5% of pixels, which makes the term
    uniform, and a uniform weight only SCALES the loss. Every intervention that
    merely scaled this mask has measured as doing nothing to its selectivity,
    so a weight that cannot discriminate is not worth adding.
    """
    k = (ferr.amax(dim=1) <= thresh).float().unsqueeze(1)
    return torch.nn.functional.interpolate(k, size=(hs, ws), mode='nearest')[:, 0]


def losses(out, gt, edge, ghost=None, floor=None, keep=None, w_edge=3.0,
           w_grad=1.0, w_ghost=6.0, w_keep=4.0):
    l1 = (out - gt).abs().mean()
    norm = edge.flatten(1).mean(dim=1).clamp_min(1e-6).view(-1, 1, 1)
    en = (edge / norm).clamp(0, 8)
    ew = 1.0 + w_edge * en
    if ghost is not None:
        # Extra weight only where BOTH hold: there is an edge, and there is a
        # real misalignment on it.
        #
        # The product is the point. Weighting edges alone is what the net
        # already had, and it made the net uniformly cautious -- measured mean R
        # near 0.5 everywhere, including 0.231 on genuinely misaligned edges
        # where Wronski reaches 0.021. Weighting misalignment alone would pay
        # for flat-area ghosts nobody sees. The product pays for exactly the
        # failure that is visible: detail from a wrongly placed frame laid over
        # an edge, which is what reads as thickening.
        # (1 + en), NOT en. This gating was the bug that made v6 ghost.
        #
        # `edge` comes from the GROUND TRUTH, so en marks where structure SHOULD
        # be. A ghost is structure where it should NOT be -- a wire smeared
        # across open sky -- and there en ~ 0, so an en-gated penalty vanishes at
        # exactly the pixels the artifact occupies. Worse, the keep term below
        # carries (1 + en) and so kept full strength there, which left the loss
        # actively pushing R UP wherever a ghost would land.
        #
        # A ghost is in fact MOST conspicuous against a smooth background, not
        # least. So merging damage must cost something everywhere, and more on an
        # edge -- never nothing off one.
        ew = ew + w_ghost * (1.0 + en) * ghost
    if keep is not None:
        # The complement of the ghost term, and the one that was missing.
        #
        # ghost pays for MERGING damage. Nothing paid for DISCARDING good
        # signal, and that asymmetry is the over-rejection: measured on
        # APC_1186, 100% of pixels are well aligned under realistic motion and
        # the mask still rejected about 30% of them, where Wronski rejected 8%.
        # The plain L1 does technically pay -- the target holds the detail a
        # rejected aligned pixel threw away -- but a fraction of a dB on an easy
        # burst was worth nothing beside ten on a broken one.
        #
        # Weighted by 1+en rather than en: a ghost is only visible on an edge,
        # but throwing away a well-aligned FLAT region costs real noise
        # reduction, so it must not be free away from edges.
        ew = ew + w_keep * (1.0 + en) * keep
    per = ((out - gt).abs().mean(dim=1) * ew).flatten(1).mean(dim=1)
    if floor is not None:
        # Each burst scored against ITS OWN achievable error, not in absolute
        # terms.
        #
        # Without this a burst with a 400 px runaway object contributes a loss
        # tens of times larger than a near-static one, so the gradient is
        # written almost entirely by the broken bursts and the net learns to be
        # cautious as a prior. That is visible in every measurement taken today:
        # mean R 0.58 on a STATIC burst where R=1 is correct and the gate gives
        # away 0.47 dB for nothing. Dividing by the loss R=1 would have incurred
        # makes "you lost 0.5 dB on an easy burst" and "you lost 10 dB on a hard
        # one" comparable statements, which is what reweighting the regimes was
        # a crude attempt at.
        per = per / floor.clamp_min(1e-6)
    l1e = per.mean()
    ax, ay = lum_grad(out)
    bx, by = lum_grad(gt)
    lg = (ax - bx).abs().mean() + (ay - by).abs().mean()
    return l1, l1e, lg, l1e + w_grad * lg


def r1_floor(d, edge, w_edge=3.0):
    """Per-burst edge-weighted error at R=1, detached. The scale each burst's
    own loss is measured against."""
    with torch.no_grad():
        # Ones on the MASK lattice, which is what gate.merge takes -- it does
        # the nearest upsample to the output grid itself. Built from feat rather
        # than from A (which is already at output resolution) or from Rw (which
        # the training batch does not fetch).
        f = d['feat']
        ones = torch.ones(f.shape[0], f.shape[1], f.shape[-2], f.shape[-1])
        o = gate.merge(d['A_ref'], d['B_ref'], d['A'], d['B'], ones)
        norm = edge.flatten(1).mean(dim=1).clamp_min(1e-6).view(-1, 1, 1)
        ew = 1.0 + w_edge * (edge / norm).clamp(0, 8)
        return ((o - d['gt']).abs().mean(dim=1) * ew).flatten(1).mean(dim=1)


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


def evaluate(net, val):
    """PSNR of Wronski / R=1 / the gate over the validation grid."""
    net.eval()
    rows = []
    with torch.no_grad():
        for i in range(val.n):
            d = val.batch([i])
            args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
            pg = psnr_t(gate.merge(*args, run_gate(net, d['feat'])), d['gt'])
            pw = psnr_t(gate.merge(*args, d['Rw']), d['gt'])
            p1 = psnr_t(gate.merge(*args, torch.ones_like(d['Rw'])), d['gt'])
            mr = float(run_gate(net, d['feat']).mean())
            rows.append((int(d['meta'][0][0]), float(d['meta'][0][1]),
                         pw, p1, pg, mr, float(d['Rw'].mean())))
    net.train()
    return rows


def report(rows):
    names = {0: 'jitter', 1: 'rot-only', 2: 'object', 3: 'outlier',
             4: 'parallax'}
    print('%-9s %-6s %-8s %-8s %-8s %-10s %-9s %-7s %-7s' %
          ('regime', 'sig_f', 'Wronski', 'R=1', 'gate', 'vs Wronski', 'vs R=1',
           'meanRg', 'meanRw'))
    for reg, sig, pw, p1, pg, mr, mw in sorted(rows, key=lambda r: (r[0], r[1])):
        print('%-9s %-6.2f %-8.2f %-8.2f %-8.2f %+-10.2f %+-9.2f %-7.3f %-7.3f'
              % (names[reg], sig, pw, p1, pg, pg - pw, pg - p1, mr, mw))
    a = np.array([[r[2], r[3], r[4]] for r in rows])
    print('%-9s %-6s %-8.2f %-8.2f %-8.2f %+-10.2f %+-9.2f'
          % ('MEAN', '', a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(),
             (a[:, 2] - a[:, 0]).mean(), (a[:, 2] - a[:, 1]).mean()))
    return float((a[:, 2] - a[:, 0]).mean())


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
    ap.add_argument('--w-keep', type=float, default=4.0,
                    help='weight on not discarding well-aligned signal. This '
                         'and --w-ghost ARE the over-rejection vs edge-precision '
                         'trade: at 4.0 the mask reaches within 0.02 dB of R=1 '
                         'on a static burst but its edge/real rises to 0.381 '
                         'from 0.139, i.e. it merges misaligned edges it should '
                         'reject.')
    ap.add_argument('--w-ghost', type=float, default=6.0,
                    help='weight on not merging a real misalignment on an edge')
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

    net = gate.SRGate(in_ch=a.in_ch or gate.NUM_FEATURES, coarse=not a.no_coarse)
    print('sr_gate: %d parameters, %d input channels'
          % (net.n_params(), net.convs[0].weight.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
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
        _, _, _, loss = losses(out, d['gt'], d['edge'],
                               ghost_risk(d['ferr'], out.shape[-2],
                                          out.shape[-1]),
                               r1_floor(d, d['edge']),
                               keep_value(d['ferr'], out.shape[-2],
                                          out.shape[-1]),
                               w_ghost=a.w_ghost, w_keep=a.w_keep)
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
        d = train.batch(idx, ('feat', 'A', 'B', 'A_ref', 'B_ref', 'gt',
                              'edge', 'ferr'))
        R = run_gate(net, d['feat'])
        out = gate.merge(d['A_ref'], d['B_ref'], d['A'], d['B'], R)
        l1, l1e, lg, loss = losses(out, d['gt'], d['edge'],
                                   ghost_risk(d['ferr'], out.shape[-2],
                                              out.shape[-1]),
                                   r1_floor(d, d['edge']),
                                   keep_value(d['ferr'], out.shape[-2],
                                              out.shape[-1]),
                                   w_ghost=a.w_ghost, w_keep=a.w_keep)
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
            g = report(evaluate(net, val))
            print('  [eval @ step %d] mean gain vs Wronski %+.2f dB' % (step, g),
                  flush=True)
            if g > best:
                best = g
                torch.save({'state_dict': net.state_dict(),
                            'dilations': net.dilations, 'width': gate.WIDTH,
                            'in_ch': (net.cconvs[0].weight.shape[1] // 2) if net.coarse
                         else net.convs[0].weight.shape[1],
                'coarse': net.coarse, 'pool': net.pool, 'out_temp': net.out_temp,
                'coarse_dilations': net.coarse_dilations, 'steps': step,
                            'val_gain': g}, outp)
                print('  saved (best so far)', flush=True)

    print('trained %d steps in %.0fs' % (step, time.time() - t0))
    g = report(evaluate(net, val))
    print('final mean gain vs Wronski %+.2f dB (best checkpoint %+.2f)'
          % (g, best))
    if g >= best:
        torch.save({'state_dict': net.state_dict(), 'dilations': net.dilations,
                    'width': gate.WIDTH, 'in_ch': (net.cconvs[0].weight.shape[1] // 2) if net.coarse
                         else net.convs[0].weight.shape[1],
                'coarse': net.coarse, 'pool': net.pool, 'out_temp': net.out_temp,
                'coarse_dilations': net.coarse_dilations,
                    'steps': step, 'val_gain': g}, outp)
        print('saved final to', outp)
    else:
        print('kept the earlier checkpoint at', outp)


if __name__ == '__main__':
    main()
