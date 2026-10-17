"""How well do the masks reject ROTATION misalignments? Held-out scene (APC_1211), app aligner + app mask + app merge.
Sweep 1: whole-camera rotation. Sweep 2: rotated content inside tile-sized patches. No deliberate flow corruption."""
import os, sys, math
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn import scenarios
from rnn.burst import Burst
from rnn.features import compute_features, consensus_features, full_consensus_features
from rnn.model import RobustNet
from real_protocol import load_scene, run_tp_real, SIZE
from target_protocol import combine, r_index, region_psnr, CLIP


def nets(paths):
    from real_protocol import load_net
    return {os.path.basename(os.path.dirname(p)): load_net(p) for p in paths}


def run(case, value, seed, scene, NN):
    rgb, cfa, wb, a0, b0 = scene
    rng = np.random.default_rng(seed)
    sc = scenarios.sample(rng, 'tremor')
    sc.update(n_objects=0, obj_speed=0.0, blur_prob=0.0, exposure_jitter=0.0, n_frames=5, tremor=0.8,
              noise_a=a0 * 3, noise_b=b0 * 9, exposure=1.0, rot_deg=0.0, rot_drift=0.0)
    if case == 'camera':
        sc.update(rot_deg=value, center_dist=float(rng.uniform(100, 400)))
    else:
        sc.update(n_rot_patches=6, rot_max_deg=value)
    b = Burst(sc, rng, Hr=SIZE, Wr=SIZE, bg_rgb=rgb, cfa=cfa, prewhite=wb)
    if case == 'patch':   # force the full angle (sign random) in every comparison frame
        for l in b.rot_patch_layers:
            l.rot_deg = [0.0] + [float(rng.choice([-1, 1]) * value) for _ in range(4)]
    dr = b.render(out_scale=1, seed=seed)
    raw, gt = dr['raw'].numpy(), dr['gt_flow'].numpy()
    d = run_tp_real(raw, gt, a0 * 3, b0 * 9, seed, False, cfa, wb)
    N, H, W = raw.shape
    gy, gx = r_index(H, W, H // 2, W // 2)
    T = combine(d['Aref'], d['Bref'], d['Ag'], d['Bg'], torch.ones(N - 1, H // 2, W // 2), gy, gx)
    Ir = d['ref_grey'][None].expand(N - 1, -1, -1).contiguous()
    feats, J = compute_features(Ir, d['grey'], d['flow'], d['r0'], a0 * 3 / 4, b0 * 9 / 4, CLIP, max(1, d['tile'] // 2))
    if os.environ.get('RNN_FULLCONS', '0') == '1':
        feats = torch.cat([feats, full_consensus_features(torch.from_numpy(raw), d['flow'], d['r0'], a0 * 3, b0 * 9)], 1)
    else:
        feats = torch.cat([feats, consensus_features(d['ref_grey'], J, d['r0'], a0 * 3 / 4, b0 * 9 / 4)], 1)
    masks = {'analytic': d['r0']}
    with torch.no_grad():
        for k, n in NN.items():
            masks[k] = n(feats[:, :n.c1.in_channels], d['r0'])[0]
    # harmful = including the frame moves the merge away from the target (other frames at analytic mask)
    harm = torch.zeros_like(d['r0'], dtype=torch.bool)
    for f in range(N - 1):
        Rw, Ro = d['r0'].clone(), d['r0'].clone(); Rw[f] = 1; Ro[f] = 0
        ew = ((combine(d['Aref'], d['Bref'], d['Au'], d['Bu'], Rw, gy, gx) - T) ** 2).mean(-1)
        eo = ((combine(d['Aref'], d['Bref'], d['Au'], d['Bu'], Ro, gy, gx) - T) ** 2).mean(-1)
        rel = F.avg_pool2d((ew - eo)[None, None], 4)[0, 0] / (F.avg_pool2d(((ew + eo) / 2)[None, None], 4)[0, 0] + 1e-9)
        harm[f] = rel > 0.10
    res = {}
    full = torch.ones(T.shape[:2], dtype=torch.bool)
    for k, R in masks.items():
        O = combine(d['Aref'], d['Bref'], d['Au'], d['Bu'], R, gy, gx)
        res[k] = (float(R[harm].mean()) if harm.any() else float('nan'), float(R[~harm].mean()), region_psnr(O, T, full))
    O1 = combine(d['Aref'], d['Bref'], d['Au'], d['Bu'], torch.ones_like(d['r0']), gy, gx)
    res['merge_all'] = (1.0, 1.0, region_psnr(O1, T, full))
    return res


def main():
    torch.set_num_threads(4)
    NN = nets(sys.argv[1:])
    scene = load_scene('APC_1211')
    for case, values in (('camera', [0.5, 1.0, 2.0, 3.0]), ('patch', [5, 10, 20, 33])):
        print(f"\n== {'whole-camera rotation (deg)' if case == 'camera' else 'rotated content inside tiles (deg)'} ==")
        print(f"{'angle':>6s} | " + " | ".join(f"{k:>22s}" for k in ['merge_all', 'analytic'] + list(NN)))
        print(f"{'':>6s} | " + " | ".join(f"{'PSNR  w_harm w_help':>22s}" for _ in range(2 + len(NN))))
        for v in values:
            acc = {}
            for s in range(3):
                for k, r in run(case, v, 4242 + s + int(v * 10), scene, NN).items():
                    acc.setdefault(k, []).append(r)
            row = []
            for k in ['merge_all', 'analytic'] + list(NN):
                a = np.array(acc[k])
                row.append(f"{np.nanmean(a[:, 2]):6.2f} {np.nanmean(a[:, 0]):6.2f} {np.nanmean(a[:, 1]):6.2f}")
            print(f"{v:6.1f} | " + " | ".join(f"{x:>22s}" for x in row), flush=True)


if __name__ == '__main__':
    main()
