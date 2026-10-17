"""Ceiling: per burst, optimise a free per-pixel mask (logits per guide pixel and frame) directly against
the perfect-alignment target through the app's merge. No mask can do better on that burst."""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
from real_protocol import RP, loss_of, _first, _wi
from target_protocol import combine, region_psnr


def main():
    torch.set_num_threads(4)
    dl = DataLoader(RP('data/rp_val', False), 1, num_workers=6, collate_fn=_first, worker_init_fn=_wi)
    rows = {}
    for d in dl:
        tag = 'corrupted' if d['corrupt'] else 'aligner-only'
        z = torch.logit(d['r0'].clamp(0.02, 0.98)).clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=0.2)
        for it in range(150):
            O = combine(d['Aref'], d['Bref'], d['A'], d['B'], torch.sigmoid(z), d['gy'], d['gx'])
            loss = ((O - d['T']) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for k, R in (('app_r0', d['r0']), ('best mask', torch.sigmoid(z))):
                O = combine(d['Aref'], d['Bref'], d['A'], d['B'], R, d['gy'], d['gx'])
                r = rows.setdefault((tag, k), {'clean': [], 'bad': [], 'all': []})
                r['clean'].append(region_psnr(O, d['T'], ~d['corr']))
                r['bad'].append(region_psnr(O, d['T'], d['corr']))
                r['all'].append(region_psnr(O, d['T'], torch.ones_like(d['corr'])))
    for tag in ('aligner-only', 'corrupted'):
        print(f"[{tag}]")
        for (t, k), r in rows.items():
            if t == tag:
                print(f"  {k:10s} clean {np.nanmean(r['clean']):6.2f}  bad {np.nanmean(r['bad']):6.2f}  whole image {np.nanmean(r['all']):6.2f}")


if __name__ == '__main__':
    main()
