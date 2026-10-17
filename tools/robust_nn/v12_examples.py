"""Example panels for the final v12 on held-out bursts (top: merges perfect|analytic|v12, bottom: masks target|analytic|v12)."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['RNN_GEO'] = '1'
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter
from real_protocol import RP, load_net, stretch
from target_protocol import combine

OUT = 'C:/Users/valen/Downloads/robust_nn_outputs/v12_examples'


def panel(d, R12, oy, ox, G, k, title):
    s = float(np.percentile(d['T'].numpy(), 99.5))
    y0, x0, S = oy * 4, ox * 4, G * 4
    tone = lambda x: (np.clip(x[y0:y0 + S, x0:x0 + S].numpy() / s, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    up = lambda a: np.kron(a, np.ones((2, 2, 1), np.uint8))
    sep = np.full((2 * S, 6, 3), 255, np.uint8)
    Ms = [d['T'], combine(d['Aref'], d['Bref'], d['A'], d['B'], d['r0'], d['gy'], d['gx']),
          combine(d['Aref'], d['Bref'], d['A'], d['B'], R12, d['gy'], d['gx'])]
    row1 = np.concatenate(sum([[up(tone(v)), sep] for v in Ms], [])[:-1], 1)
    def mk(m):
        a = (m[k, oy:oy + G, ox:ox + G].clamp(0, 1).numpy() * 255).astype(np.uint8)
        return np.repeat(np.kron(a, np.ones((8, 8), np.uint8))[..., None], 3, 2)
    row2 = np.concatenate(sum([[mk(m), sep] for m in (d['geo'], d['r0'], R12)], [])[:-1], 1)
    head = np.full((28, row1.shape[1], 3), 255, np.uint8)
    img = Image.fromarray(np.concatenate([head, row1, np.full((6, row1.shape[1], 3), 255, np.uint8), row2], 0))
    dr = ImageDraw.Draw(img)
    dr.text((6, 7), title, fill=(0, 0, 0))
    return img


def main():
    torch.set_num_threads(3)
    os.makedirs(OUT, exist_ok=True)
    net = load_net('runs/v12/model.pt')
    ds = RP('data/rp12_val', False)
    cand = []
    for i in list(range(30, 60)) + list(range(0, 30, 3)):
        d = ds[i]
        for kk in ('A', 'B', 'Aref', 'Bref', 'T'):
            d[kk] = d[kk].float()
        with torch.no_grad():
            R12 = stretch(net(d['feats'], d['r0'])[0])
        T = d['T'].mean(-1).numpy()
        gy_, gx_ = np.gradient(T)
        Hg, Wg = d['r0'].shape[-2:]
        edge = uniform_filter(gy_ ** 2 + gx_ ** 2, 4)[::4, ::4][:Hg, :Wg]
        bad = (d['geo'] < 0.2).float().sum(0).numpy()
        sc = uniform_filter(bad * edge / (edge.mean() + 1e-12), 24)
        sc[:12] = sc[-12:] = 0; sc[:, :12] = sc[:, -12:] = 0
        gy0, gx0 = np.unravel_index(np.argmax(sc), sc.shape)
        G = 48
        oy, ox = int(min(max(0, gy0 - G // 2), Hg - G)), int(min(max(0, gx0 - G // 2), Wg - G))
        t = d['geo']; reg = (slice(None), slice(oy, oy + G), slice(ox, ox + G))
        k = int((t[reg] < 0.2).float().sum((1, 2)).argmax())
        tb, tg = t < 0.2, t > 0.8
        stats = dict(a_leak=float(d['r0'][tb].mean()), a_keep=float(d['r0'][tg].mean()),
                     v_leak=float(R12[tb].mean()), v_keep=float(R12[tg].mean()))
        cand.append((sc.max(), i, oy, ox, G, k, stats, d, R12))
    cand.sort(key=lambda c: -c[0])
    used, n = set(), 0
    for sc, i, oy, ox, G, k, st, d, R12 in cand:
        if i in used:
            continue
        used.add(i); n += 1
        title = (f"burst {i}  frame {k + 1}   whole burst: analytic leak {st['a_leak']:.2f} keep {st['a_keep']:.2f}  |  "
                 f"v12 leak {st['v_leak']:.2f} keep {st['v_keep']:.2f}")
        panel(d, R12, oy, ox, G, k, title).save(os.path.join(OUT, f'example_{n:02d}_burst{i}.png'))
        print(f'example {n}: {title}')
        if n >= 8:
            break


if __name__ == '__main__':
    main()
