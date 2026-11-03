import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['RNN_GEO'] = '1'
from PIL import Image, ImageDraw
from real_protocol import RP, load_net, stretch

OUT = 'C:/Users/valen/Downloads/robust_nn_outputs/v12_accuracy_examples'


def main():
    torch.set_num_threads(2)
    os.makedirs(OUT, exist_ok=True)
    net = load_net('runs/v12/model.pt')
    ds = RP('data/rp13_val', False)
    tot = [0, 0]
    for n, i in enumerate([30, 33, 36, 40, 44, 48, 52, 3]):
        d = ds[i]
        with torch.no_grad():
            R = stretch(net(d['feats'], d['r0'])[0])
        t = d['geo']
        bad, good = t < 0.2, t > 0.8
        k = int(bad.float().sum((1, 2)).argmax())                    # the frame with most misalignment
        tk, Rk = t[k].numpy(), R[k].numpy()
        b, g = tk < 0.2, tk > 0.8
        ok = (b & (Rk < 0.5)) | (g & (Rk > 0.5))
        acc = ok.sum() / max(1, (b | g).sum())
        tot[0] += ok.sum(); tot[1] += (b | g).sum()
        agree = np.full(tk.shape + (3,), 128, np.uint8)
        agree[g & (Rk > 0.5)] = (255, 255, 255)
        agree[b & (Rk < 0.5)] = (0, 0, 0)
        agree[b & (Rk >= 0.5)] = (230, 30, 30)                      # leak: misaligned but kept
        agree[g & (Rk <= 0.5)] = (40, 90, 230)                      # over-rejection: good but dropped
        ref = d['T'].float().numpy()[::4, ::4]
        s = np.percentile(ref, 99.5)
        refi = (np.clip(ref / s, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        gray = lambda m: np.repeat((np.clip(m, 0, 1) * 255).astype(np.uint8)[..., None], 3, 2)
        tiles = [gray(tk), gray(Rk), agree, refi[:tk.shape[0], :tk.shape[1]]]
        up = lambda a: np.kron(a, np.ones((3, 3, 1), np.uint8))
        sp = np.full((tk.shape[0] * 3, 8, 3), 255, np.uint8)
        row = np.concatenate(sum([[up(x), sp] for x in tiles], [])[:-1], 1)
        img = Image.fromarray(np.concatenate([np.full((26, row.shape[1], 3), 255, np.uint8), row], 0))
        nb, ng = int(b.sum()), int(g.sum())
        ImageDraw.Draw(img).text((6, 6), f"held-out burst {i}, frame {k + 1}:  accuracy {acc:.1%}   "
                                          f"(misaligned px kept/red: {int((b & (Rk >= 0.5)).sum())}/{nb}, good px dropped/blue: "
                                          f"{int((g & (Rk <= 0.5)).sum())}/{ng})   [GT mask | v12 mask | agreement | image]",
                                 fill=(0, 0, 0))
        img.save(f'{OUT}/example_{n + 1:02d}_burst{i}_frame{k + 1}.png')
        print(f'example {n + 1}: burst {i} frame {k + 1} accuracy {acc:.3f}')
    print(f'these frames together: {tot[0] / tot[1]:.3f}')


if __name__ == '__main__':
    main()
