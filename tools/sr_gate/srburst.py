"""Burst synthesis from real Bayer DNG, with an oracle ground truth.

A trained mask needs a target, and the target has to be the MERGED image -- the
project already measured that R* = 1/(1 + delta^2/sigma^2) style labels
systematically over-reject in the sub-pixel band, because a sub-pixel offset is
the signal the SR merge feeds on rather than damage. So:

  scene S        a real DNG demosaiced at full sensor resolution (4032x3024),
                 treated as the latent high-resolution radiance. Linear, camera
                 native, white balanced -- the same domain raw_io hands the
                 pipeline after prewhitening.
  frame n        S resampled at HALF resolution through a known motion, CFA
                 mosaicked, and given Poisson-Gaussian noise at the DNG's own
                 alpha/beta times a gain. This is the raw the pipeline sees.
  ground truth   the merge run on the NOISE-FREE frames with the TRUE per-pixel
                 flow and R == 1 everywhere.

That ground truth is the point. It is what the merge itself would produce given
perfect alignment, so it does not reward sharpness the reconstruction kernel
cannot deliver, and it does not punish the network for a blur that is the
kernel's doing. What is left in the residual is exactly what R is responsible
for: correspondences that should not have been merged, and correspondences that
should have been.

Alignment error is synthesised rather than measured, over the regimes that
matter and in their real proportions:
  * per-tile jitter, swept from 0.1 px (deep in the band where rejection is
    harmful) to 6 px (registration failure),
  * a translation-only estimate of a ROTATING scene, which is coherent
    within-tile error -- the mode a residual test cannot see,
  * an independently moving object whose tiles lock onto the background,
    which is the ghost case,
  * gross per-tile outliers.
"""
from __future__ import annotations

import glob
import os

import numpy as np

import srsim
from srsim import CFA, SCALE, Cfg


# --------------------------------------------------------------------------
# scene loading
# --------------------------------------------------------------------------

# Scene sources that are NOT under `root`. bursts/ and the hdrplus payloads sit
# one directory above the repo; `ours` is one above that again, so no single
# relative pattern reaches both. Override or extend with SR_GATE_SCENES, an
# os.pathsep separated list of directories to glob for *.dng.
# ours, ours2, ours3, ... -- one 8-frame handheld burst of one scene each. Globbed
# rather than listed so another capture is picked up by dropping the folder in.
# sorted() inside find_dngs keeps the order stable, which the holdout depends on.
EXTRA_SCENE_GLOBS = ['../ours*/*.dng']


def find_dngs(root, extra=True):
    """Real Bayer DNGs only. photos-dng-test/ and ok/ hold linear demosaiced
    outputs, which are not sensor data and are excluded.

    Order is stable -- sorted within each pattern, patterns in a fixed order, and
    the out-of-tree sources LAST. make_data.py draws its holdout from a
    permutation of this list, so anything that reorders it silently changes which
    scenes are held out and makes new numbers incomparable with old ones.
    """
    pats = [
        'bursts/**/*.dng',
        'hdrplus-python-main/test_data/**/*.dng',
    ]
    out = []
    for p in pats:
        out += sorted(glob.glob(os.path.join(root, p), recursive=True))
    n_in_tree = len(out)
    if extra:
        for p in EXTRA_SCENE_GLOBS:
            out += sorted(glob.glob(os.path.join(root, p), recursive=True))
        for d in os.environ.get('SR_GATE_SCENES', '').split(os.pathsep):
            if d.strip():
                out += sorted(glob.glob(os.path.join(d.strip(), '*.dng')))
    out = [p for p in out if 'photos-dng-test' not in p.replace(chr(92), '/')]
    # Deduplicate while keeping order, in case a glob and SR_GATE_SCENES overlap.
    seen = set()
    uniq = []
    for p in out:
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq, n_in_tree


def load_scene(path):
    """Demosaic a DNG at full sensor resolution into a linear, white balanced,
    camera-native RGB scene in [0, 1]. No gamma, no auto brightness, no colour
    space conversion -- the app's raw is prewhitened camera native, and the
    scene has to live in the same domain for the noise model to mean anything.

    Returns None for a file that is not real Bayer sensor data.
    """
    import rawpy
    try:
        r = rawpy.imread(path)
        if r.raw_pattern is None or r.num_colors != 3:
            return None
        rgb = r.postprocess(
            gamma=(1, 1), no_auto_bright=True, output_bps=16,
            use_camera_wb=True, output_color=rawpy.ColorSpace.raw,
            no_auto_scale=False, half_size=False,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            median_filter_passes=0,
        )
    except Exception:
        return None
    s = rgb.astype(np.float32) / 65535.0
    # The scene is the LATENT signal for a burst we then add noise to, so pull
    # the level down a little to leave headroom and keep the synthetic noise
    # from clipping against the white point.
    return np.clip(s * 0.92, 0.0, 1.0)


# --------------------------------------------------------------------------
# motion
# --------------------------------------------------------------------------

class Motion:
    """A similarity in RAW pixel coordinates mapping REFERENCE -> frame n,
    which is the direction the merge's flow points (lr_mov = lr + flow)."""

    def __init__(self, theta, tx, ty, cy, cx):
        self.theta = float(theta)
        self.t = np.array([tx, ty], np.float64)
        self.c = np.array([cx, cy], np.float64)

    def forward(self, x, y):
        ct, st = np.cos(self.theta), np.sin(self.theta)
        dx = x - self.c[0]
        dy = y - self.c[1]
        return (ct * dx - st * dy + self.c[0] + self.t[0],
                st * dx + ct * dy + self.c[1] + self.t[1])

    def inverse(self, x, y):
        ct, st = np.cos(-self.theta), np.sin(-self.theta)
        dx = x - self.c[0] - self.t[0]
        dy = y - self.c[1] - self.t[1]
        return (ct * dx - st * dy + self.c[0],
                st * dx + ct * dy + self.c[1])


def _bilinear(img, y, x):
    """Bilinear sample of an HxWxC image, edge clamped."""
    h, w = img.shape[:2]
    y = np.clip(y, 0, h - 1)
    x = np.clip(x, 0, w - 1)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (y - y0)[..., None]
    fx = (x - x0)[..., None]
    top = img[y0, x0] + (img[y0, x1] - img[y0, x0]) * fx
    bot = img[y1, x0] + (img[y1, x1] - img[y1, x0]) * fx
    return top + (bot - top) * fy


def _box2_blur(s):
    """The sensor aperture: a 2x2 box at HR scale, i.e. one raw pixel's
    footprint. Deliberately no more than that -- a stronger PSF would remove
    the aliasing that is the whole reason multi-frame SR works, and the gate
    has to learn when that aliasing is safe to merge."""
    p = np.pad(s, ((0, 1), (0, 1), (0, 0)), mode='edge')
    return 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])


# --------------------------------------------------------------------------
# burst synthesis
# --------------------------------------------------------------------------

class BurstSpec:
    def __init__(self, rng, regime=None):
        r = rng
        self.regime = regime if regime is not None else r.integers(0, 4)
        # handheld shake: a few raw px of translation, up to ~0.6 degrees
        self.trans = float(r.uniform(0.3, 6.0))
        self.theta = float(r.uniform(0.0, 0.010)) * (1 if r.random() < 0.7 else 0)
        # per-tile estimate jitter, log-uniform across the whole regime range
        self.sigma_flow = float(np.exp(r.uniform(np.log(0.08), np.log(6.0))))
        # a translation-only estimate of a rotating field: coherent, within-tile
        self.translation_only = bool(self.regime == 1)
        if self.translation_only:
            self.theta = float(r.uniform(0.002, 0.012))
            self.sigma_flow = float(np.exp(r.uniform(np.log(0.08), np.log(0.8))))
        # independently moving object
        self.has_object = bool(self.regime == 2)
        self.obj_vel = float(r.uniform(1.5, 10.0))
        self.obj_lock_p = float(r.uniform(0.5, 1.0))
        # gross outlier tiles
        self.outlier_p = float(r.uniform(0.0, 0.06)) if self.regime == 3 else 0.0
        self.noise_gain = float(np.exp(r.uniform(np.log(1.0), np.log(24.0))))
        self.n_frames = 8


def synth_burst(scene, spec: BurstSpec, rng, tile_size=16):
    """Returns a dict with the raw frames, the estimated per-tile flow, the true
    per-pixel flow at output resolution, and the noise-free frames."""
    Hs, Ws = scene.shape[:2]
    h, w = Hs // SCALE, Ws // SCALE
    blurred = _box2_blur(scene)

    cy, cx = h / 2.0, w / 2.0
    N = spec.n_frames
    motions = [Motion(0.0, 0.0, 0.0, cy, cx)]
    obj_motions = [Motion(0.0, 0.0, 0.0, cy, cx)]
    for n in range(1, N):
        f = n / (N - 1.0)
        th = spec.theta * f * rng.uniform(0.6, 1.4)
        tx = spec.trans * f * rng.uniform(-1, 1)
        ty = spec.trans * f * rng.uniform(-1, 1)
        motions.append(Motion(th, tx, ty, cy, cx))
        if spec.has_object:
            ox = tx + spec.obj_vel * f * rng.uniform(-1, 1)
            oy = ty + spec.obj_vel * f * rng.uniform(-1, 1)
            obj_motions.append(Motion(th, ox, oy, cy, cx))
        else:
            obj_motions.append(motions[-1])

    # object support, in REFERENCE raw coordinates
    obj_mask = np.zeros((h, w), bool)
    if spec.has_object:
        oh = int(rng.uniform(0.25, 0.55) * h)
        ow = int(rng.uniform(0.25, 0.55) * w)
        oy0 = int(rng.uniform(0, h - oh))
        ox0 = int(rng.uniform(0, w - ow))
        obj_mask[oy0:oy0 + oh, ox0:ox0 + ow] = True

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cfa_ch = CFA[np.arange(h)[:, None] & 1, np.arange(w)[None, :] & 1]

    raws = []
    raws_clean = []
    for n in range(N):
        pbx, pby = motions[n].inverse(xx, yy)
        if spec.has_object:
            pox, poy = obj_motions[n].inverse(xx, yy)
            inside = ((poy >= 0) & (poy < h) & (pox >= 0) & (pox < w))
            oi = np.clip(np.rint(poy).astype(np.int64), 0, h - 1)
            oj = np.clip(np.rint(pox).astype(np.int64), 0, w - 1)
            is_obj = inside & obj_mask[oi, oj]
            px = np.where(is_obj, pox, pbx)
            py = np.where(is_obj, poy, pby)
        else:
            px, py = pbx, pby
        samp = _bilinear(blurred, py * SCALE, px * SCALE)
        clean = np.take_along_axis(samp, cfa_ch[..., None], axis=2)[..., 0]
        clean = np.clip(clean, 0.0, 1.0).astype(np.float32)
        var = np.maximum(srsim.ALPHA_DNG * spec.noise_gain * clean +
                         srsim.BETA_DNG * spec.noise_gain ** 2, 0.0)
        noisy = np.clip(clean + rng.standard_normal(clean.shape) * np.sqrt(var),
                        0.0, 1.0).astype(np.float32)
        raws_clean.append(clean)
        raws.append(noisy)

    # --- TRUE per-pixel flow at output resolution (for the ground truth merge)
    Hso, Wso = h * SCALE, w * SCALE
    hi, hj = np.mgrid[0:Hso, 0:Wso]
    lr_y = hi / SCALE
    lr_x = hj / SCALE
    obj_here = obj_mask[np.clip(np.rint(lr_y).astype(np.int64), 0, h - 1),
                        np.clip(np.rint(lr_x).astype(np.int64), 0, w - 1)]
    true_flow = []
    for n in range(N):
        fx_b, fy_b = motions[n].forward(lr_x, lr_y)
        if spec.has_object:
            fx_o, fy_o = obj_motions[n].forward(lr_x, lr_y)
            fx = np.where(obj_here, fx_o, fx_b)
            fy = np.where(obj_here, fy_o, fy_b)
        else:
            fx, fy = fx_b, fy_b
        true_flow.append((fx - lr_x, fy - lr_y))

    # --- ESTIMATED per-tile flow (what block matching would hand the mask)
    ts = tile_size
    ny, nx = h // ts, w // ts
    tcy = (np.arange(ny) + 0.5) * ts
    tcx = (np.arange(nx) + 0.5) * ts
    TCY, TCX = np.meshgrid(tcy, tcx, indexing='ij')
    tile_obj = np.zeros((ny, nx), bool)
    if spec.has_object:
        for i in range(ny):
            for j in range(nx):
                blk = obj_mask[i * ts:(i + 1) * ts, j * ts:(j + 1) * ts]
                tile_obj[i, j] = blk.mean() > 0.5

    flows = []
    for n in range(N):
        bx, by = motions[n].forward(TCX, TCY)
        fx = bx - TCX
        fy = by - TCY
        if spec.translation_only:
            # the estimate collapses the rotating field to its mean translation
            fx = np.full_like(fx, fx.mean())
            fy = np.full_like(fy, fy.mean())
        if spec.has_object:
            ox, oy = obj_motions[n].forward(TCX, TCY)
            lock = tile_obj & (rng.random((ny, nx)) < spec.obj_lock_p)
            fx = np.where(tile_obj & ~lock, ox - TCX, fx)
            fy = np.where(tile_obj & ~lock, oy - TCY, fy)
        if n > 0:
            fx = fx + rng.standard_normal((ny, nx)) * spec.sigma_flow
            fy = fy + rng.standard_normal((ny, nx)) * spec.sigma_flow
            if spec.outlier_p > 0:
                o = rng.random((ny, nx)) < spec.outlier_p
                fx = np.where(o, fx + rng.uniform(-12, 12, (ny, nx)), fx)
                fy = np.where(o, fy + rng.uniform(-12, 12, (ny, nx)), fy)
        else:
            fx = np.zeros_like(fx)
            fy = np.zeros_like(fy)
        flows.append(np.stack([fx, fy], axis=-1).astype(np.float32))

    return dict(raws=raws, raws_clean=raws_clean, flows=flows,
                true_flow=true_flow, obj_mask=obj_mask, tile_size=ts,
                noise_gain=spec.noise_gain, regime=spec.regime,
                sigma_flow=spec.sigma_flow, h=h, w=w)
