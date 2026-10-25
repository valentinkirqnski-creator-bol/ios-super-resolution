"""Shared fixtures for the tests: small synthetic bursts with known motion."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np          # noqa: E402
import torch                # noqa: E402

from mfsr import formation, scenes     # noqa: E402
from rnn import scenarios as rsc       # noqa: E402

torch.set_num_threads(4)


def world(rng, n=4, tremor=1.5, rot=0.0, zoom=0.0, objects=0, optics=0.4, exposure=0.6, highlights=False):
    sc = rsc.sample(rng, "tremor")
    sc.update(n_frames=n, n_objects=objects, obj_speed=4.0 if objects else 0.0, obj_parallax=(1.0, 1.5),
              blur_prob=0.0, exposure_jitter=0.0, rot_deg=rot, rot_drift=0.0, zoom_rate=zoom, persp=0.0,
              cam_drift=0.0, highlights=highlights, optics_sigma=optics, exposure=exposure, tremor=tremor,
              center_dist=50.0)
    return sc


def burst(seed=0, S=128, scene="APC_1340", aperture=3, iso=2.0, **kw):
    rng = np.random.default_rng(seed)
    rgb, cfa, wb, a0, b0 = scenes.load_real(scene)
    sc = world(rng, **kw)
    sc.update(noise_a=a0 * iso, noise_b=b0 * iso * iso)
    f = formation.Formation(sc, rng, S, 2.0, rgb, cfa, wb, aperture, 1.0, obj_src=rgb if kw.get("objects") else None)
    return f, cfa, wb, sc
