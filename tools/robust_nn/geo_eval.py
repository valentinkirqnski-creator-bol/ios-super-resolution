"""Score masks the user's way, against the geometric misalignment target:
   leak = weight on truly misaligned samples (target < 0.2), ideal 0
   keep = weight on correctly aligned samples (target > 0.8), ideal 1
   AUC  = ranking of misaligned below aligned (1.0 = perfect)
Outputs are scored as shipped: raw net R, and stretched (exact 0/1 below 0.05 / above 0.95)."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['RNN_GEO'] = '1'
from torch.utils.data import DataLoader
from real_protocol import RP, _first, _wi, load_net, stretch
from auc_check import auc


def main():
    torch.set_num_threads(3)
    nets = {os.path.basename(os.path.dirname(p)): load_net(p) for p in sys.argv[1:]}
    dl = DataLoader(RP(os.environ.get('VAL', 'data/rp12_val'), False), 1, num_workers=int(os.environ.get('NW', '3')), collate_fn=_first,
                    worker_init_fn=_wi, prefetch_factor=1)
    rng = np.random.default_rng(0)
    acc = {}
    for d in dl:
        t = d['geo']
        masks = {'analytic': d['r0']}
        with torch.no_grad():
            for k, n in nets.items():
                R = n(d['feats'][:, :n.c1.in_channels], d['r0'])[0]
                masks[k] = R
                masks[k + ' (exact 0/1)'] = stretch(R)
        bad, good = t < 0.2, t > 0.8
        sel = torch.from_numpy(rng.random(t.shape) < 0.3)
        for k, R in masks.items():
            a = acc.setdefault(k, {'leak': [], 'keep': [], 'nb': [], 'pg': []})
            a['leak'].append(float(R[bad].sum())); a['keep'].append(float(R[good].sum()))
            a['nb'].append(R[bad & sel].numpy()); a['pg'].append(R[good & sel].numpy())
            a.setdefault('cnt', [0, 0]); a['cnt'][0] += int(bad.sum()); a['cnt'][1] += int(good.sum())
    print(f"{'mask':22s} {'leak (ideal 0)':>15s} {'keep (ideal 1)':>15s} {'AUC':>7s}")
    for k, a in acc.items():
        print(f"{k:22s} {sum(a['leak']) / a['cnt'][0]:15.3f} {sum(a['keep']) / a['cnt'][1]:15.3f} "
              f"{auc(np.concatenate(a['nb']), np.concatenate(a['pg'])):7.3f}")


if __name__ == '__main__':
    main()
