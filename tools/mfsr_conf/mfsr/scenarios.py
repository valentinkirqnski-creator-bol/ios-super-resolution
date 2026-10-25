"""Evaluation categories (and the training mix). Each draws a world (rnn.burst parameters), a scene source, the
alignment the 'estimated' variant uses, and the deliberate corruption of the 'corrupted' variant.

Every burst is stored with TWO alignment variants over the SAME rendered frames: v0 = the estimate (the app's aligner,
or the exact tile vectors where the category is about correctly aligned data) and v1 = v0 + deliberate errors. The
network therefore sees the same pixels with good and with bad correspondences, and only the evidence separates them.

  aligned_texture  real scene, sub-pixel hand tremor (pure translation), exact tile vectors: every sample is valid
  aliasing         fine text / bar charts, light optics so the sensor aliases, tremor, exact tile vectors (situation A)
  edge_slip        real scene, app aligner, small slips of edge content (0.1-0.8 px, scene motion the tile flow cannot
                   follow) plus edge-targeted / burst-consistent flow errors in v1 (situation B, subtle)
  repetitive       bar charts or real scene with drift, app aligner (wrong-period locks), large / block errors
  flat_noisy       low light, high ISO: noise must not read as motion; errors in flat areas are harmless (situation C)
  rotation         camera roll 0.3-2.5 deg (+ slow zoom) about an off-centre point; app aligner with or without the
                   global homography, or exact tile-centre vectors (tile-centre-correct, tile-edge-wrong)
  moving           1-3 independently moving objects (cut from the scene) with occlusion and motion blur
  exposure         per-frame exposure changes 5-30%, matched by the app's gain with a 2-6% residual error
  saturation       clipped highlights
"""
import math

import numpy as np

from rnn import scenarios as rsc

from .corrupt import TYPES

CATEGORIES = ["aligned_texture", "aliasing", "edge_slip", "repetitive", "flat_noisy", "rotation", "moving", "exposure",
              "saturation"]


def _logu(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def make(category, rng, cfg):
    """-> dict(sc, scene_kind 'real'|'text'|'chart', tex, align0 'oracle'|'aligner'|'homog', kinds, p_frame,
    edge_burst, exp_residual)."""
    sc = rsc.sample(rng, "tremor")
    f0, f1 = cfg["frames"]
    lo, hi = cfg["iso_mult"]
    sc.update(n_frames=int(rng.integers(f0, f1 + 1)), n_objects=0, obj_speed=0.0, blur_prob=0.0, exposure_jitter=0.0,
              rot_deg=0.0, rot_drift=0.0, zoom_rate=0.0, persp=0.0, cam_drift=0.0, highlights=False,
              optics_sigma=float(rng.uniform(0.25, 0.6)), exposure=_logu(rng, 0.35, 1.1),
              tremor=float(rng.uniform(0.3, 3.0)), center_dist=float(rng.uniform(0, 400)))
    iso = _logu(rng, lo, hi)
    out = dict(sc=sc, scene_kind="real", tex=cfg["tex_per_raw"], align0="aligner", kinds=list(TYPES),
               p_frame=cfg["corrupt_prob"], edge_burst=False, exp_residual=cfg["exposure_match_residual"], iso=iso)
    if category == "aligned_texture":
        out.update(align0="oracle")
    elif category == "aliasing":
        sc.update(optics_sigma=float(rng.uniform(0.2, 0.4)), exposure=float(rng.uniform(0.35, 0.8)),
                  tremor=float(rng.uniform(0.5, 3.0)))
        out.update(scene_kind="text" if rng.random() < 0.6 else "chart", tex=cfg["tex_per_raw_procedural"],
                   align0="oracle", iso=_logu(rng, 1.0, 8.0))
    elif category == "edge_slip":
        sc.update(tremor=float(rng.uniform(0.3, 2.5)))
        if rng.random() < 0.6:   # content slips on straight edges (rnn/burst.py n_rot_patches_small / small_disp)
            sc.update(n_rot_patches_small=int(rng.integers(2, 9)), small_disp=(0.1, 0.8), small_anywhere=0.2,
                      small_pframe=0.7, small_size=(24.0, 96.0))
        out.update(kinds=["edge", "tile_subpix", "smooth"], edge_burst=rng.random() < 0.5,
                   align0="aligner" if rng.random() < 0.7 else "oracle")
    elif category == "repetitive":
        sc.update(tremor=float(rng.uniform(0.5, 4.0)), cam_drift=float(rng.uniform(0.0, 3.0)))
        if rng.random() < 0.6:
            out.update(scene_kind="chart", tex=cfg["tex_per_raw_procedural"])
        out.update(kinds=["tile_large", "block", "tile_subpix"])
    elif category == "flat_noisy":
        sc.update(exposure=_logu(rng, 0.03, 0.12), tremor=float(rng.uniform(0.3, 2.5)))
        out.update(iso=_logu(rng, 10.0, 60.0), align0="aligner" if rng.random() < 0.7 else "oracle")
    elif category == "rotation":
        sc.update(rot_deg=float(rng.uniform(0.3, 2.5)), rot_drift=float(np.radians(rng.uniform(-0.3, 0.3))),
                  center_dist=float(rng.uniform(0, 600)),
                  zoom_rate=float(rng.uniform(0.0, 0.004)) if rng.random() < 0.5 else 0.0)
        u = rng.random()
        out.update(align0="aligner" if u < 0.5 else ("homog" if u < 0.8 else "oracle"),
                   kinds=["rotation", "tile_subpix", "smooth", "edge"])
    elif category == "moving":
        sc.update(n_objects=int(rng.integers(1, 4)), obj_speed=float(rng.uniform(1.0, 10.0)),
                  obj_rot_deg=float(rng.uniform(0, 3.0)), obj_scale_rate=float(rng.uniform(0, 0.02)),
                  obj_jitter=float(rng.uniform(0, 1.0)), obj_parallax=(1.0, 2.0), blur_prob=0.2,
                  exposure_frac=float(rng.uniform(0.2, 0.5)))
        out.update(p_frame=0.4)
    elif category == "exposure":
        sc.update(exposure_jitter=float(rng.uniform(0.05, 0.3)))
        out.update(exp_residual=float(rng.uniform(0.02, 0.06)), align0="aligner" if rng.random() < 0.6 else "oracle")
    elif category == "saturation":
        sc.update(highlights=True, exposure=float(rng.uniform(0.9, 2.0)))
    else:
        raise ValueError(category)
    return out


def pick(rng, weights):
    names = list(weights)
    p = np.array([weights[n] for n in names], np.float64)
    return names[int(rng.choice(len(names), p=p / p.sum()))]
