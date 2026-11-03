"""Per-pixel ground truth of 'actual misalignment': for each comparison frame k and output pixel, does
including frame k (R_k = 1 vs 0, other frames at the app's mask) move the APP merge toward or away from the
perfect-alignment target? Then: what weight do the app mask and the NN give to harmful vs helpful samples."""
import os, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
from real_protocol import RP, _first, _wi, load_net
from target_protocol import combine
from rnn.model import RobustNet


def main():
    torch.set_num_threads(3)
    net = load_net(sys.argv[1])
    dl = DataLoader(RP('data/rp_val', False), 1, num_workers=3, collate_fn=_first, worker_init_fn=_wi, prefetch_factor=1)
    acc = {}
    for d in dl:
        for k in ('A', 'B', 'Aref', 'Bref', 'T'):
            d[k] = d[k].float()
        with torch.no_grad():
            Rn = net(d['feats'][:, :net.c1.in_channels], d['r0'])[0]
        r0 = d['r0']
        K = r0.shape[0]
        for k in range(K):
            Rw, Ro = r0.clone(), r0.clone()
            Rw[k] = 1.0
            Ro[k] = 0.0
            ew = ((combine(d['Aref'], d['Bref'], d['A'], d['B'], Rw, d['gy'], d['gx']) - d['T']) ** 2).mean(-1)
            eo = ((combine(d['Aref'], d['Bref'], d['A'], d['B'], Ro, d['gy'], d['gx']) - d['T']) ** 2).mean(-1)
            # to the guide grid (output is 4x guide: scale 2 x quad 2)
            delta = F.avg_pool2d((ew - eo)[None, None], 4)[0, 0]          # >0: including frame k hurts
            scale = F.avg_pool2d(((ew + eo) / 2)[None, None], 4)[0, 0] + 1e-9
            rel = delta / scale
            harm = rel > 0.10
            help_ = rel < -0.02
            fe = d['ferr'][k]
            for name, m in (('flow err <0.1px', fe < 0.1), ('flow err 0.1-1px', (fe >= 0.1) & (fe < 1)),
                            ('flow err >1px', fe >= 1), ('all', torch.ones_like(fe, dtype=torch.bool))):
                for cls, cm in (('harmful', harm), ('helpful', help_)):
                    mm = m & cm
                    a = acc.setdefault((name, cls), [0.0, 0.0, 0])
                    a[0] += float(r0[k][mm].sum()); a[1] += float(Rn[k][mm].sum()); a[2] += int(mm.sum())
    print(f"{'pixels':18s} {'class':8s} {'count':>9s} | {'app mask w':>10s} {'v5 w':>7s}")
    for (name, cls), (s0, s1, n) in acc.items():
        if n:
            print(f"{name:18s} {cls:8s} {n:9d} | {s0 / n:10.3f} {s1 / n:7.3f}")


if __name__ == '__main__':
    main()
