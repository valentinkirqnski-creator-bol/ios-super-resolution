"""Synthetic burst from the car scene of burst9 (APC_1186) with deliberate misalignments and rotated tile
content, through the app's aligner / analytic mask / merge: GT output + GT mask, analytic output + mask,
v12 output + mask (exact 0/1)."""
import os, sys, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageDraw
from rnn import scenarios
from rnn.burst import Burst
from rnn.features import compute_features
from real_protocol import load_scene, run_tp_real, load_net, stretch, geo_target, SIZE, CLIP
from target_protocol import combine, r_index

OUT = 'C:/Users/valen/Downloads/robust_nn_outputs/car_synthetic'


def main():
    torch.set_num_threads(3)
    os.makedirs(OUT, exist_ok=True)
    rgb, cfa, wb, a0, b0 = load_scene('APC_1186')
    # car front / plate of burst9: raw (2156, 1852) -> scene texel (quad) coords
    cx, cy, half = 1078, 926, 576            # texture needs 2*(SIZE+192) = 1152 texels
    scene = np.ascontiguousarray(rgb[cy - half:cy + half, cx - half:cx + half])
    rng = np.random.default_rng(11)
    sc = scenarios.sample(rng, 'tremor')
    sc.update(n_objects=0, obj_speed=0.0, blur_prob=0.0, exposure_jitter=0.0, n_frames=6, tremor=1.2,
              noise_a=a0 * 3, noise_b=b0 * 9, exposure=1.0, n_rot_patches=6, n_rot_patches_edge=5, rot_max_deg=33.0,
              persp=1e-7)
    b = Burst(sc, rng, Hr=SIZE, Wr=SIZE, bg_rgb=scene, cfa=cfa, prewhite=wb, allow_flip=False)
    d = b.render(out_scale=1, seed=5)
    raw, gt = d['raw'].numpy(), d['gt_flow'].numpy()
    rc = d['raw_clean'].numpy(); g = np.empty_like(rc)
    for i in range(2):
        for j in range(2):
            g[:, i::2, j::2] = rc[:, i::2, j::2] / wb[cfa[i][j]]
    clean = torch.from_numpy(0.25 * (g[:, 0::2, 0::2] + g[:, 0::2, 1::2] + g[:, 1::2, 0::2] + g[:, 1::2, 1::2]))
    o = run_tp_real(raw, gt, a0 * 3, b0 * 9, 77, True, cfa, wb)        # app aligner + deliberate corruption
    N, H, W = raw.shape
    gy, gx = r_index(H, W, H // 2, W // 2)
    ones = torch.ones(N - 1, H // 2, W // 2)
    # ground truth uses only samples that truly exist: visible in that frame and not at a motion discontinuity
    # (where the merge's flow interpolation would mix the two motions)
    vis = d['vis'][1:].float()
    gtf = torch.from_numpy(gt[1:])
    jump = torch.zeros_like(vis)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        jump = torch.maximum(jump, (gtf - torch.roll(gtf, (dy, dx), (2, 3))).norm(dim=1))
    Rgt = vis * (jump < 0.25).float()
    T = combine(o['Aref'], o['Bref'], o['Ag'], o['Bg'], Rgt, gy, gx)
    T_old = combine(o['Aref'], o['Bref'], o['Ag'], o['Bg'], ones, gy, gx)
    geo = geo_target(clean, o['flow'], torch.from_numpy(gt[1:]), d['vis'][1:].float(), a0 * 3, b0 * 9)
    Ir = o['ref_grey'][None].expand(N - 1, -1, -1).contiguous()
    feats, _ = compute_features(Ir, o['grey'], o['flow'], o['r0'], a0 * 3 / 4, b0 * 9 / 4, CLIP, max(1, o['tile'] // 2))
    with torch.no_grad():
        R12 = stretch(load_net('runs/v12/model.pt')(feats, o['r0'])[0])
    outs = {'1_ground_truth': T, '1b_old_ground_truth_all_samples': T_old,
            '2_analytic': combine(o['Aref'], o['Bref'], o['Au'], o['Bu'], o['r0'], gy, gx),
            '3_v12': combine(o['Aref'], o['Bref'], o['Au'], o['Bu'], R12, gy, gx),
            '0_merge_everything': combine(o['Aref'], o['Bref'], o['Au'], o['Bu'], ones, gy, gx)}
    s = float(np.percentile(T.numpy(), 99.5))
    gain = np.median(T.numpy().reshape(-1, 3), 0); gain = gain[1] / gain
    tone = lambda x: (np.clip(x.numpy() * gain / s, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    for k, v in outs.items():
        Image.fromarray(tone(v)).save(f'{OUT}/output_{k}.png')
    masks = {'1_ground_truth': geo, '2_analytic': o['r0'], '3_v12': R12}
    bad = (geo < 0.2).float().sum((1, 2)); k = int(bad.argmax())
    for kk, m in masks.items():
        Image.fromarray((m.mean(0).clamp(0, 1).numpy() * 255).astype(np.uint8)).resize((2 * W, 2 * H), Image.NEAREST).save(
            f'{OUT}/mask_{kk}_average_all_frames.png')
        Image.fromarray((m[k].clamp(0, 1).numpy() * 255).astype(np.uint8)).resize((2 * W, 2 * H), Image.NEAREST).save(
            f'{OUT}/mask_{kk}_frame{k + 1}.png')
    # one overview sheet: outputs top, frame-k masks bottom
    T_, A_, V_ = tone(outs['1_ground_truth']), tone(outs['2_analytic']), tone(outs['3_v12'])
    mk = lambda m: np.repeat(np.asarray(Image.fromarray((m[k].clamp(0, 1).numpy() * 255).astype(np.uint8)).resize(
        (2 * W, 2 * H), Image.NEAREST))[..., None], 3, 2)
    sp = np.full((2 * H, 10, 3), 255, np.uint8)
    top = np.concatenate([T_, sp, A_, sp, V_], 1)
    bot = np.concatenate([mk(geo), sp, mk(o['r0']), sp, mk(R12)], 1)
    sheet = Image.fromarray(np.concatenate([np.full((30, top.shape[1], 3), 255, np.uint8), top,
                                            np.full((10, top.shape[1], 3), 255, np.uint8), bot], 0))
    ImageDraw.Draw(sheet).text((8, 9), f"top: ground truth | analytic | v12 (exact 0/1)    bottom: masks for frame {k + 1}"
                                         f" (white = keep, black = reject): ground truth | analytic | v12", fill=(0, 0, 0))
    sheet.save(f'{OUT}/overview.png')
    tb, tg = geo < 0.2, geo > 0.8
    dec = tb | tg
    for nm, m in (('analytic', o['r0']), ('v12', R12)):
        acc = float((((tb) & (m < 0.5)) | ((tg) & (m > 0.5))).sum() / dec.sum())
        print(f"{nm:9s} accuracy vs GT mask {acc:.3f}   leak {float(m[tb].mean()):.3f}   keep {float(m[tg].mean()):.3f}")
    print('saved to', OUT, ' frame shown:', k + 1)


if __name__ == '__main__':
    main()
