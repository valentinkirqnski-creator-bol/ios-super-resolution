"""Does the mask RANK harmful samples below helpful ones? AUC = P(w_harmful < w_helpful), ties count half.
Unaffected by uniform darkening. Harmful/helpful defined through the app merge vs the perfect target."""
import os, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
from real_protocol import RP, _first, _wi, load_net
from target_protocol import combine
from rnn.model import RobustNet


def auc(neg, pos):
    """neg = weights on harmful (should be low), pos = weights on helpful. Mann-Whitney U."""
    x = np.concatenate([neg, pos]); lab = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    order = np.argsort(x, kind='mergesort'); xs = x[order]
    ranks = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    rp = ranks[lab == 1].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    torch.set_num_threads(3)
    nets = {}
    for p in sys.argv[1:]:
        nets[os.path.basename(os.path.dirname(p))] = load_net(p)
    dl = DataLoader(RP(os.environ.get('VAL', 'data/rp_val'), False), 1, num_workers=3, collate_fn=_first, worker_init_fn=_wi, prefetch_factor=1)
    rng = np.random.default_rng(0)
    W = {k: {'harm': [], 'help': [], 'harm_sub': [], 'help_sub': []} for k in ['analytic'] + list(nets)}
    for d in dl:
        for k in ('A', 'B', 'Aref', 'Bref', 'T'):
            d[k] = d[k].float()
        masks = {'analytic': d['r0']}
        with torch.no_grad():
            for k, n in nets.items():
                masks[k] = n(d['feats'][:, :n.c1.in_channels], d['r0'])[0]
        r0 = d['r0']
        for f in range(r0.shape[0]):
            Rw, Ro = r0.clone(), r0.clone(); Rw[f] = 1; Ro[f] = 0
            ew = ((combine(d['Aref'], d['Bref'], d['A'], d['B'], Rw, d['gy'], d['gx']) - d['T']) ** 2).mean(-1)
            eo = ((combine(d['Aref'], d['Bref'], d['A'], d['B'], Ro, d['gy'], d['gx']) - d['T']) ** 2).mean(-1)
            rel = F.avg_pool2d((ew - eo)[None, None], 4)[0, 0] / (F.avg_pool2d(((ew + eo) / 2)[None, None], 4)[0, 0] + 1e-9)
            harm, help_ = (rel > 0.10).flatten().numpy(), (rel < -0.02).flatten().numpy()
            sub = ((d['ferr'][f] >= 0.1) & (d['ferr'][f] < 1)).flatten().numpy()      # subtle misalignment zone
            sel = rng.random(harm.shape) < 0.25                                       # subsample for speed
            for k, R in masks.items():
                w = R[f].flatten().numpy()
                W[k]['harm'].append(w[harm & sel]); W[k]['help'].append(w[help_ & sel])
                W[k]['harm_sub'].append(w[harm & sel & sub]); W[k]['help_sub'].append(w[help_ & sel & sub])
    print(f"{'mask':10s} {'AUC all':>8s} {'AUC subtle 0.1-1px':>19s}   (0.5 = no knowledge, 1.0 = perfect)")
    for k, v in W.items():
        c = {kk: np.concatenate(vv) for kk, vv in v.items()}
        print(f"{k:10s} {auc(c['harm'], c['help']):8.3f} {auc(c['harm_sub'], c['help_sub']):19.3f}")


if __name__ == '__main__':
    main()
