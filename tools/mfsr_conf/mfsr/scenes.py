"""Scene sources.

Real scenes: one DNG each, decoded by the app's own loader (dng_dump.exe -> tools/robust_nn/data/real/<name>.f32),
and every RGGB/BGGR quad turned into ONE camera-native linear RGB texel (the quad's four sites, white balance
undone). No demosaicing and no resampling: each texel is a real colour measurement of the scene at the source's
quad resolution (2016 x 1512 for a 12 MP iPhone frame).

The simulated sensor samples that scene at `tex_per_raw` texels per sensor pixel (default 2). At the 2x merge scale an
output pixel then covers exactly one texel, so the HR target is resolved by genuine scene data, never by upscaling.
What this does NOT give is the real phone's absolute scale: a simulated sensor pixel spans 2 source texels = 4 source
pixels, i.e. the synthetic frames are a quarter-resolution view of the source scene. No 48 MP captures exist on this
machine (the iPhone 15 cannot record 48 MP RAW), so the 12 MP DNGs are the highest-resolution real sources available.

Procedural scenes (aliasing, repetitive patterns): text and bar charts rendered directly at `tex_per_raw_procedural`
texels per sensor pixel (tools/robust_nn/rnn/text_scene.py, chart_scene.py), sharp by construction.

Splits are BY SOURCE IMAGE (configs: scenes.train / val / test), so no crop of a validation or test scene can appear
in training. The references of the real test bursts (burst9 = APC_1186, burst12 = APC_1294) are excluded entirely.
"""
import os

import numpy as np

from . import ROBUST_NN

REAL = os.path.join(ROBUST_NN, "data", "real")
_CACHE = {}


def load_real(name):
    """-> (rgb [h,w,3] float32 camera-native linear, cfa ((c00,c01),(c10,c11)), wb [3], alpha, beta)."""
    if name in _CACHE:
        return _CACHE[name]
    with open(os.path.join(REAL, name + ".meta")) as fh:
        meta = fh.read().split()
    h, w = int(meta[0]), int(meta[1])
    cfa = ((int(meta[2]), int(meta[3])), (int(meta[4]), int(meta[5])))
    a, b = float(meta[6]), float(meta[7])
    wb = np.array([float(meta[8]), float(meta[9]), float(meta[10])], np.float32)
    r = np.fromfile(os.path.join(REAL, name + ".f32"), np.float32).reshape(h, w)
    r = r[:h // 2 * 2, :w // 2 * 2]
    q = {}
    for i in range(2):
        for j in range(2):
            q.setdefault(cfa[i][j], []).append(r[i::2, j::2])
    rgb = np.stack([np.mean(q[c], 0) / wb[c] for c in range(3)], -1)
    out = (np.clip(rgb, 0, None).astype(np.float32), cfa, wb, a, b)
    _CACHE[name] = out
    return out


def procedural(kind, rng, n_tex, tex_per_raw, wb):
    """Square procedural scene of n_tex texels: 'text' (aliasing fine text) or 'chart' (bars / repetitive)."""
    if kind == "text":
        from rnn.text_scene import text_scene
        return text_scene(rng, n_tex, n_tex, tex_per_raw, wb, cap_raw=(1.8, 7.0))
    from rnn.chart_scene import chart_scene
    return chart_scene(rng, n_tex, n_tex, tex_per_raw, wb, (2.0, 10.0))
