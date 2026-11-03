"""Plain-dB answer per region on held-out bursts (rp7_val):
  well-aligned regions : output vs the perfect-alignment target (higher = closer to perfect)
  misaligned regions   : output vs the target AND vs the reference-only merge (does it fall back to the reference?)"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
from real_protocol import RP, _first, _wi, load_net
from target_protocol import combine, region_psnr


def main():
    torch.set_num_threads(2)
    nets = {os.path.basename(os.path.dirname(p)): load_net(p) for p in sys.argv[1:]}
    dl = DataLoader(RP('data/rp7_val', False), 1, num_workers=2, collate_fn=_first, worker_init_fn=_wi, prefetch_factor=1)
    acc = {}
    for d in dl:
        for k in ('A', 'B', 'Aref', 'Bref', 'T'):
            d[k] = d[k].float()
        R = {'merge everything': torch.ones_like(d['r0']), 'reference only': torch.zeros_like(d['r0']), 'analytic': d['r0']}
        with torch.no_grad():
            for k, n in nets.items():
                R[k] = n(d['feats'][:, :n.c1.in_channels], d['r0'])[0]
        Oref = combine(d['Aref'], d['Bref'], d['A'], d['B'], R['reference only'], d['gy'], d['gx'])
        good, bad = ~d['corr'], d['corr']
        for k, r in R.items():
            O = combine(d['Aref'], d['Bref'], d['A'], d['B'], r, d['gy'], d['gx'])
            a = acc.setdefault(k, [[], [], []])
            a[0].append(region_psnr(O, d['T'], good)); a[1].append(region_psnr(O, d['T'], bad)); a[2].append(region_psnr(O, Oref, bad))
    print(f"{'mask':17s} {'well-aligned: vs perfect':>25s} {'misaligned: vs perfect':>23s} {'misaligned: vs reference-only':>30s}")
    for k, (g, b, r) in acc.items():
        print(f"{k:17s} {np.nanmean(g):25.2f} {np.nanmean(b):23.2f} {np.nanmean(r):30.2f}")


if __name__ == '__main__':
    main()
