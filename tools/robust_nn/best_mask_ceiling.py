"""Ceiling: per burst, solve directly for the best possible mask (a free weight per guide pixel and frame)
against the perfect-alignment target, through the app's merge. No network can beat it on these bursts."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
from real_protocol import RP, _first, _wi, load_net
from target_protocol import combine, region_psnr


def main():
    torch.set_num_threads(4)
    v7 = load_net('runs/v7/model.pt')
    ds = RP('data/rp7_val', False)
    ds = torch.utils.data.Subset(ds, list(range(0, len(ds), 2)))
    dl = DataLoader(ds, 1, num_workers=3, collate_fn=_first, worker_init_fn=_wi, prefetch_factor=1)
    rows = {}
    for d in dl:
        for k in ('A', 'B', 'Aref', 'Bref', 'T'):
            d[k] = d[k].float()
        with torch.no_grad():
            r7 = v7(d['feats'][:, :14], d['r0'])[0]
        z = torch.logit(r7.clamp(0.02, 0.98)).clone().requires_grad_(True)       # start from v7
        opt = torch.optim.Adam([z], lr=0.3)
        for it in range(60):
            O = combine(d['Aref'], d['Bref'], d['A'], d['B'], torch.sigmoid(z), d['gy'], d['gx'])
            loss = ((O - d['T']) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        masks = {'merge everything': torch.ones_like(d['r0']), 'reference only': torch.zeros_like(d['r0']),
                 'analytic': d['r0'], 'v7': r7, 'BEST POSSIBLE': torch.sigmoid(z).detach()}
        full = torch.ones(d['T'].shape[:2], dtype=torch.bool)
        with torch.no_grad():
            for k, R in masks.items():
                O = combine(d['Aref'], d['Bref'], d['A'], d['B'], R, d['gy'], d['gx'])
                r = rows.setdefault(k, [[], [], []])
                r[0].append(region_psnr(O, d['T'], ~d['corr'])); r[1].append(region_psnr(O, d['T'], d['corr']))
                r[2].append(region_psnr(O, d['T'], full))
    print(f"{'mask':17s} {'well-aligned dB':>16s} {'misaligned dB':>14s} {'whole image dB':>15s}   (vs the perfect-alignment merge)")
    for k, (g, b, a) in rows.items():
        print(f"{k:17s} {np.nanmean(g):16.2f} {np.nanmean(b):14.2f} {np.nanmean(a):15.2f}")


if __name__ == '__main__':
    main()
