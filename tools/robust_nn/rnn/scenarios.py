"""Scenario sampler. Weighted so the common handheld cases (where an over-eager
mask destroys super-resolution) are not drowned out by easy obvious motion.

    tremor       normal hand tremor: small, mostly correctly-estimated sub-pixel
                 shifts -> must be KEPT (aliasing differs, correspondence right)
    rotation     small camera roll about an off-crop centre: locally wrong
                 tile-translation model even though global alignment looks good
    zoom         walking toward the subject: depth-dependent scale (parallax)
    parallax     lateral camera translation over layers at different depths
    moving       independently moving subjects, occlusion / disocclusion
    repetitive   bricks / text / stripes: plausible-looking wrong matches
    low_light    high ISO flat regions: noise must not read as motion
    saturated    clipped highlights: agreement is weak evidence
    motion_blur  smeared frames / subjects
"""
import numpy as np

WEIGHTS = {
    "tremor": 0.20, "rotation": 0.11, "zoom": 0.09, "parallax": 0.11, "moving": 0.14,
    "repetitive": 0.11, "low_light": 0.10, "saturated": 0.07, "motion_blur": 0.07,
}
NAMES = list(WEIGHTS)


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample(rng, name=None):
    if name is None:
        p = np.array([WEIGHTS[n] for n in NAMES])
        name = NAMES[rng.choice(len(NAMES), p=p / p.sum())]
    a = _logu(rng, 5e-5, 3e-3)
    sc = dict(
        name=name, n_frames=int(rng.integers(4, 9)), texture_mode="mixed", highlights=False,
        tremor=rng.uniform(0.4, 3.0), cam_drift=0.0, rot_deg=0.01, rot_drift=0.0,
        center_dist=rng.uniform(0, 400), zoom_rate=0.0, persp=_logu(rng, 1e-7, 3e-6),
        n_objects=int(rng.random() < 0.3), obj_parallax=(1.0, 1.15), obj_speed=0.0,
        obj_rot_deg=0.0, obj_scale_rate=0.0, obj_jitter=0.0,
        blur_prob=0.08, blur_px=rng.uniform(0.5, 2.0), exposure_frac=0.3,
        optics_sigma=rng.uniform(0.35, 0.9), exposure=_logu(rng, 0.25, 1.0),
        exposure_jitter=0.0 if rng.random() < 0.6 else rng.uniform(0.0, 0.04),
        noise_a=a, noise_b=a * _logu(rng, 1e-3, 2e-2),
        corrupt_prob=0.45,
    )
    if name == "tremor":
        sc.update(tremor=rng.uniform(0.3, 2.5), corrupt_prob=0.35)
    elif name == "rotation":
        deg = rng.uniform(0.2, 1.5)
        sc.update(rot_deg=deg, center_dist=rng.uniform(100, min(1500, 30 / np.radians(deg))),
                  rot_drift=np.radians(rng.normal(0, 0.15)))
    elif name == "zoom":
        sc.update(zoom_rate=rng.uniform(0.003, 0.015), n_objects=int(rng.integers(1, 4)),
                  obj_parallax=(1.2, 3.0), center_dist=rng.uniform(0, 300))
    elif name == "parallax":
        sc.update(cam_drift=rng.uniform(1.5, 7.0), n_objects=int(rng.integers(1, 4)), obj_parallax=(1.3, 4.0))
    elif name == "moving":
        sc.update(n_objects=int(rng.integers(1, 4)), obj_speed=rng.uniform(1.0, 12.0),
                  obj_rot_deg=rng.uniform(0, 3.0), obj_scale_rate=rng.uniform(0, 0.02),
                  obj_jitter=rng.uniform(0, 1.0), obj_parallax=(1.0, 2.0))
    elif name == "repetitive":
        sc.update(texture_mode="repetitive", n_objects=int(rng.integers(0, 3)),
                  obj_speed=rng.uniform(0, 6.0) if rng.random() < 0.5 else 0.0,
                  cam_drift=rng.uniform(0, 4.0), obj_parallax=(1.0, 2.0))
    elif name == "low_light":
        a = _logu(rng, 1.5e-3, 1e-2)
        sc.update(exposure=_logu(rng, 0.04, 0.2), noise_a=a, noise_b=a * _logu(rng, 2e-3, 3e-2),
                  n_objects=int(rng.integers(0, 3)), obj_speed=rng.uniform(0, 6.0))
    elif name == "saturated":
        sc.update(highlights=True, exposure=rng.uniform(0.7, 1.5), n_objects=int(rng.integers(0, 3)),
                  obj_speed=rng.uniform(0, 5.0) if rng.random() < 0.5 else 0.0)
    elif name == "motion_blur":
        sc.update(blur_prob=rng.uniform(0.4, 0.9), blur_px=rng.uniform(1.0, 6.0),
                  n_objects=int(rng.integers(0, 3)), obj_speed=rng.uniform(0, 8.0), exposure_frac=rng.uniform(0.2, 0.6))
    return sc
