"""AUC (harmful vs helpful ranking) + per-region dB for the analytic mask, v7 and the pixel CNN, same bursts."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
from train_pixel import PixDS, PixelNet
from real_protocol import load_net, _first, _wi
from auc_check import auc
from target_protocol import combine, region_psnr


def main():
    torch.set_num_threads(3)
    v7 = load_net('runs/v7/model.pt')
    pn = PixelNet(); pn.load_state_dict(torch.load(sys.argv[1], map_location='cpu')); pn.eval()
    dl = DataLoader(PixDS(os.environ.get('VAL', 'data/rp7_val'), False), 1, num_workers=3, collate_fn=_first,
                    worker_init_fn=_wi, prefetch_factor=1)
    W = {k: {'harm': [], 'help': [], 'harm_s': [], 'help_s': [], 'good': [], 'bad': []} for k in ('analytic', 'v7', 'v11')}
    for d in dl:
        for k in ('A', 'B', 'Aref', 'Bref', 'T'):
            d[k] = d[k].float()
        with torch.no_grad():
            m = {'analytic': d['r0'], 'v7': v7(d['feats'][:, :14], d['r0'])[0], 'v11': pn(d['pix'], d['r0'])[0]}
        lab = d['lab']; sub = (d['ferr'] >= 0.1) & (d['ferr'] < 1)
        for k, R in m.items():
            W[k]['harm'].append(R[lab == 0].numpy()); W[k]['help'].append(R[lab == 1].numpy())
            W[k]['harm_s'].append(R[(lab == 0) & sub].numpy()); W[k]['help_s'].append(R[(lab == 1) & sub].numpy())
            O = combine(d['Aref'], d['Bref'], d['A'], d['B'], R, d['gy'], d['gx'])
            W[k]['good'].append(region_psnr(O, d['T'], ~d['corr'])); W[k]['bad'].append(region_psnr(O, d['T'], d['corr']))
    print(f"{'mask':9s} {'AUC all':>8s} {'AUC subtle':>11s} | {'w harmful':>9s} {'w helpful':>9s} | {'well-aligned dB':>15s} {'misaligned dB':>13s}")
    for k, v in W.items():
        c = {kk: np.concatenate(vv) for kk, vv in v.items() if kk in ('harm', 'help', 'harm_s', 'help_s')}
        sub_h = np.random.default_rng(0)
        print(f"{k:9s} {auc(c['harm'], c['help']):8.3f} {auc(c['harm_s'], c['help_s']):11.3f} | {c['harm'].mean():9.3f} {c['help'].mean():9.3f} | "
              f"{np.nanmean(v['good']):15.2f} {np.nanmean(v['bad']):13.2f}")


if __name__ == '__main__':
    main()
