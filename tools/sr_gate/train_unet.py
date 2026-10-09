"""Supervised robustness: a pseudo-ground-truth mask, plus the merge loss.

Two problems with training SRGate against the merged image alone.

AMBIGUITY. R -> merge(R) -> L(merge, GT) is an inverse problem with many
solutions. With seven good frames the target is reproduced about as well by
R = [1,1,1,1,1,1,1] as by [1,1,0.7,1,0.8,1,1] or [1,0.5,1,1,0.4,1,1], or by any
number of spatially varying mixtures. The merge loss only knows whether the
final image resembled the target; it does not know that one particular
robustness field was wanted, so it cannot uniquely identify it.

NO DIRECT SUPERVISION, although the labels are free. The synthetic pipeline
knows the true flow, so the actual alignment error

    dF_n = F_hat_n - F_star_n

is available per pixel (build.flow_error stores |dF| on the mask lattice). That
turns robustness into an ordinary supervised problem.

The target is soft rather than a 0/1 threshold:

    R* = exp(-|dF|^2 / (2 sigma_F^2))        sigma_F = 0.2 px

which reproduces the intended response:

    flow error   R*            flow error   R*
      0.00 px    1.00            0.30 px    0.33
      0.05 px    0.97            0.50 px    0.04
      0.10 px    0.88

Both losses are kept, because they answer different questions:

    L_mask  = |R - R*|            did you identify bad alignment?
    L_merge = |I_out - I_GT|      did that decision matter to the picture?

plus the edge-weighted and gradient terms already in train.py, and a smoothness
term on R:

    L = w_mask L_mask + w_merge L_merge + w_edge L_edge + w_grad L_grad
        + w_smooth L_smooth

    python make_data.py --out data_un --train 110 --val 25
    python pack.py --data data_un
    python train_unet.py --data data_un --out sr_gate_unet.pt
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
import train as T

SIGMA_F = 0.2


def pseudo_gt(ferr, sigma_f=SIGMA_F):
    """R* = exp(-|dF|^2 / (2 sigma_F^2)), the soft geometric-safety target."""
    return torch.exp(-(ferr ** 2) / (2.0 * sigma_f * sigma_f))


def run_unet(net, feat, img):
    """(b, N, 8, h, w) + (b, N, 12, h, w) -> R (b, N, h, w)."""
    x = torch.cat([feat[:, :, :8], img], dim=2)
    b, n = x.shape[0], x.shape[1]
    r = net(x.flatten(0, 1))[:, 0]
    return r.view(b, n, x.shape[-2], x.shape[-1])


def smooth_loss(R):
    """Total variation on the mask. The reliability field is physically smooth
    except at motion boundaries, and nothing else in the loss discourages
    speckle -- an isolated deep rejection costs almost no merged error."""
    dx = (R[..., :, 1:] - R[..., :, :-1]).abs().mean()
    dy = (R[..., 1:, :] - R[..., :-1, :]).abs().mean()
    return dx + dy


def losses(R, Rstar, out, gt, edge, w):
    l_mask = (R - Rstar).abs().mean()
    l_merge = (out - gt).abs().mean()
    norm = edge.flatten(1).mean(dim=1).clamp_min(1e-6).view(-1, 1, 1)
    ew = 1.0 + 3.0 * (edge / norm).clamp(0, 8)
    l_edge = ((out - gt).abs().mean(dim=1) * ew).mean()
    ax, ay = T.lum_grad(out)
    bx, by = T.lum_grad(gt)
    l_grad = (ax - bx).abs().mean() + (ay - by).abs().mean()
    l_smooth = smooth_loss(R)
    total = (w['mask'] * l_mask + w['merge'] * l_merge + w['edge'] * l_edge +
             w['grad'] * l_grad + w['smooth'] * l_smooth)
    return total, l_mask, l_merge, l_edge, l_grad, l_smooth


def grad_norms(net, d, sigma_f, w):
    """||dL_term/dtheta|| for each term separately.

    Raw loss magnitudes do not say which term is actually steering the weights:
    a large flat term can contribute less gradient than a small sharp one. Run
    at eval time only, since it costs one backward pass per term.
    """
    R = run_unet(net, d['feat'], d['img'])
    Rs = pseudo_gt(d['ferr'], sigma_f)
    out = gate.merge(d['A_ref'], d['B_ref'], d['A'], d['B'], R)
    norm = d['edge'].flatten(1).mean(dim=1).clamp_min(1e-6).view(-1, 1, 1)
    ew = 1.0 + 3.0 * (d['edge'] / norm).clamp(0, 8)
    ax, ay = T.lum_grad(out)
    bx, by = T.lum_grad(d['gt'])
    terms = {
        'mask': w['mask'] * (R - Rs).abs().mean(),
        'merge': w['merge'] * (out - d['gt']).abs().mean(),
        'edge': w['edge'] * ((out - d['gt']).abs().mean(dim=1) * ew).mean(),
        'grad': w['grad'] * ((ax - bx).abs().mean() + (ay - by).abs().mean()),
        'smooth': w['smooth'] * smooth_loss(R),
    }
    ps = [p for p in net.parameters() if p.requires_grad]
    out_s = {}
    for k, v in terms.items():
        g = torch.autograd.grad(v, ps, retain_graph=True, allow_unused=True)
        out_s[k] = float(torch.sqrt(sum((x * x).sum() for x in g
                                        if x is not None)))
    return out_s


def evaluate(net, val, base, sigma_f=SIGMA_F):
    rows = []
    net.eval()
    with torch.no_grad():
        for i in range(val.n):
            d = val.batch([i])
            args = (d['A_ref'], d['B_ref'], d['A'], d['B'])
            R = run_unet(net, d['feat'], d['img'])
            Rs = pseudo_gt(d['ferr'], sigma_f)
            Rb = T.run_gate(base, d['feat'])
            rows.append((
                int(d['meta'][0][0]), float(d['meta'][0][1]),
                float((R - Rs).abs().mean()), float((Rb - Rs).abs().mean()),
                T.psnr_t(gate.merge(*args, R), d['gt']),
                T.psnr_t(gate.merge(*args, Rb), d['gt']),
                float(R.mean()), float(Rb.mean()), float(Rs.mean()),
                # Spatial variation: a mask that collapsed to a constant has
                # std 0, whatever its mean. Reported so "did it collapse" is
                # answered directly instead of inferred from the mean.
                float(R.std()), float(Rs.std())))
    net.train()
    return rows


def report(rows):
    names = {0: 'jitter', 1: 'rot-only', 2: 'object', 3: 'outlier',
             4: 'parallax'}
    print('%-9s %-6s %-9s %-9s %-8s %-8s %-7s %-7s %-7s %-7s %-7s' %
          ('regime', 'sig_f', '|R-R*|', 'base|R-R*|', 'PSNR', 'base', 'meanR',
           'base', 'meanR*', 'stdR', 'stdR*'))
    for r in sorted(rows, key=lambda q: (q[0], q[1])):
        print('%-9s %-6.2f %-9.4f %-9.4f  %-8.2f %-8.2f %-7.3f %-7.3f %-7.3f '
              '%-7.3f %-7.3f'
              % (names[r[0]], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
                 r[9], r[10]))
    v = np.array([[r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]]
                  for r in rows])
    print('%-9s %-6s %-9.4f %-9.4f  %-8.2f %-8.2f %-7.3f %-7.3f %-7.3f %-7.3f '
          '%-7.3f'
          % ('MEAN', '', v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(),
             v[:, 3].mean(), v[:, 4].mean(), v[:, 5].mean(), v[:, 6].mean(),
             v[:, 7].mean(), v[:, 8].mean()))
    # Selected on agreement with the pseudo-ground-truth mask: lower is better,
    # so the score is negated. PSNR is printed alongside, not selected on.
    return -float(v[:, 0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_un')
    ap.add_argument('--out', default='sr_gate_unet.pt')
    ap.add_argument('--base', default='sr_gate.pt')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-every', type=float, default=420.0)
    ap.add_argument('--sigma-f', type=float, default=SIGMA_F)
    ap.add_argument('--w-mask', type=float, default=1.0)
    ap.add_argument('--w-merge', type=float, default=1.0)
    ap.add_argument('--w-edge', type=float, default=1.0)
    ap.add_argument('--w-grad', type=float, default=1.0)
    ap.add_argument('--w-smooth', type=float, default=0.05)
    ap.add_argument('--base-ch', type=int, default=16)
    ap.add_argument('--mid-ch', type=int, default=28)
    ap.add_argument('--head-bias', type=float, default=0.0,
                    help='logit the head opens at. 0 = R 0.5, the '
                         'maximum-gradient point.')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    w = {'mask': a.w_mask, 'merge': a.w_merge, 'edge': a.w_edge,
         'grad': a.w_grad, 'smooth': a.w_smooth}

    here = os.path.dirname(os.path.abspath(__file__))
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    train = T.Split(root, 'train')
    val = T.Split(root, 'val')
    base = gate.from_checkpoint(torch.load(
        a.base if os.path.isabs(a.base) else os.path.join(here, a.base),
        map_location='cpu', weights_only=False))

    net = gate.SRGateUNet(base=a.base_ch, mid=a.mid_ch,
                          head_bias=a.head_bias)
    print('unet: %d parameters, %d input channels (8 stats + %d image), '
          'receptive field %d mask px'
          % (net.n_params(), net.in_ch, gate.IMG_CH, net.receptive_field()),
          flush=True)
    print('weights %s  sigma_F %.2f px' % (w, a.sigma_f), flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=a.steps,
                                                pct_start=0.15)
    best, t0, log_t, eval_t = -1e9, time.time(), time.time(), time.time()
    outp = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)

    for step in range(1, a.steps + 1):
        idx = sorted(rng.choice(train.n, size=a.batch, replace=False).tolist())
        d = train.batch(idx, ('feat', 'img', 'ferr', 'A', 'B', 'A_ref',
                              'B_ref', 'gt', 'edge'))
        R = run_unet(net, d['feat'], d['img'])
        Rs = pseudo_gt(d['ferr'], a.sigma_f)
        out = gate.merge(d['A_ref'], d['B_ref'], d['A'], d['B'], R)
        total, lm, lg_, le, lgr, ls = losses(R, Rs, out, d['gt'], d['edge'], w)
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if time.time() - log_t > 45.0:
            log_t = time.time()
            print('  step %5d/%d  %4.0fs  mask %.4f  merge %.4f  edge %.4f  '
                  'grad %.4f  sm %.4f  meanR %.3f/%.3f  lr %.1e'
                  % (step, a.steps, time.time() - t0, lm.item(), lg_.item(),
                     le.item(), lgr.item(), ls.item(), R.mean().item(),
                     Rs.mean().item(), sched.get_last_lr()[0]), flush=True)
        if time.time() - eval_t > a.eval_every:
            eval_t = time.time()
            sc = report(evaluate(net, val, base, a.sigma_f))
            gn = grad_norms(net, train.batch(idx, (
                'feat', 'img', 'ferr', 'A', 'B', 'A_ref', 'B_ref',
                'gt', 'edge')), a.sigma_f, w)
            print('  [eval @ step %d] grad norms  %s'
                  % (step, '  '.join('%s %.2e' % (k, v)
                                     for k, v in gn.items())),
                  flush=True)
            print('  [eval @ step %d] |R-R*| %.4f (lower better)'
                  % (step, -sc), flush=True)
            if sc > best:
                best = sc
                torch.save(gate.unet_checkpoint(net, steps=step,
                                                mask_err=-sc), outp)
                print('  saved (best so far)', flush=True)

    sc = report(evaluate(net, val, base, a.sigma_f))
    print('final |R-R*| %.4f (best %.4f)' % (-sc, -best))
    if sc >= best:
        torch.save(gate.unet_checkpoint(net, steps=a.steps, mask_err=-sc), outp)
        print('saved final to', outp)
    else:
        print('kept the earlier checkpoint at', outp)


if __name__ == '__main__':
    main()
