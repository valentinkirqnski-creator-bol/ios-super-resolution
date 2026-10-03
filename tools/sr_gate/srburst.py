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
EXTRA_SCENE_GLOBS = ['../ours*/*.dng', '../shots/*.dng', '../*.dng']


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

def _par_dir(m):
    """Unit direction of a frame's camera translation; (0,0) when it did not
    move, so a stationary frame gets no parallax rather than a random one."""
    d = m.t
    n = float(np.hypot(d[0], d[1]))
    return (0.0, 0.0) if n < 1e-9 else (float(d[0]) / n, float(d[1]) / n)


class Motion:
    """Camera motion in RAW pixel coordinates, REFERENCE -> frame n, which is
    the direction the merge's flow points (lr_mov = lr + flow).

    Roll, yaw and pitch, plus an image-plane translation.

    Roll alone is a similarity, and that is all this modelled before. Yaw and
    pitch are rotations about the camera's own vertical and horizontal axes, so
    they are PROJECTIVE: the image transform is K R K^-1, and the displacement
    they produce grows quadratically toward the frame edges rather than
    linearly. That matters here for two reasons. A per-tile constant flow
    approximates a projective field worst at the sides, which is where the
    straight-line misalignments show up; and no amount of in-plane rotation
    reproduces the keystone, so a mask trained only on roll has never seen the
    shape. Handheld capture has yaw and pitch in it as a matter of course --
    they are what pointing the phone slightly differently looks like.

    Reduces EXACTLY to the old similarity when yaw = pitch = 0: K R K^-1 for a
    pure roll is the in-plane rotation, so existing behaviour is unchanged.
    """

    def __init__(self, theta, tx, ty, cy, cx, yaw=0.0, pitch=0.0, f=None,
                 scale=1.0):
        self.theta = float(theta)
        # Dolly: the camera moving along its optical axis. A pure scale about the
        # principal point, so the flow is RADIAL and grows with distance from
        # centre -- another field a per-tile constant vector fits worst at the
        # frame edges, which is where the reported misalignments are.
        self.scale = float(scale)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.t = np.array([tx, ty], np.float64)
        self.c = np.array([cx, cy], np.float64)
        # Focal length in raw pixels. 1.5x the half-diagonal is a normal phone
        # field of view; it only sets how much keystone a given yaw produces.
        self.f = float(f) if f else 3.0 * float(max(cx, cy))

    def _R(self, inv=False):
        cr, sr = np.cos(self.theta), np.sin(self.theta)
        cy_, sy_ = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        Rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cy_, 0.0, sy_], [0.0, 1.0, 0.0], [-sy_, 0.0, cy_]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        R = Ry @ Rx @ Rz
        return R.T if inv else R

    def _warp(self, x, y, R, tx, ty, sc=1.0):
        X = (x - self.c[0]) / self.f
        Y = (y - self.c[1]) / self.f
        Xp = R[0, 0] * X + R[0, 1] * Y + R[0, 2]
        Yp = R[1, 0] * X + R[1, 1] * Y + R[1, 2]
        Zp = R[2, 0] * X + R[2, 1] * Y + R[2, 2]
        # Behind the camera cannot be imaged; clamp rather than produce a sign
        # flip, which would fold the frame onto itself.
        Zp = np.where(np.abs(Zp) < 1e-6, 1e-6, Zp)
        return (self.f * Xp / Zp * sc + self.c[0] + tx,
                self.f * Yp / Zp * sc + self.c[1] + ty)

    def forward(self, x, y):
        return self._warp(x, y, self._R(False), self.t[0], self.t[1],
                          self.scale)

    def inverse(self, x, y):
        return self._warp(x - self.t[0], y - self.t[1], self._R(True), 0.0,
                          0.0, 1.0 / max(self.scale, 1e-9))


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
    """Motion parameters, with the ranges taken from measure_motion.py rather
    than guessed.

    What the guessed ranges got wrong, measured on the real bursts:

      * GLOBAL translation was 0.3-6 raw px. Real handheld bursts here reach
        264 px between the reference and the last frame. span and emag are built
        from flow magnitudes, so the whole feature distribution was off.
      * OBJECT velocity was 1.5-10 px. The ours2 subject moves ~250 px and
        changes pose, which is the case the mask actually failed on, and no
        training burst was within a factor of 25 of it.
      * objects were axis-aligned rectangles translating rigidly, which is the
        one shape whose displaced copy still matches itself.
    """

    # Motion magnitude as a STRATIFIED LADDER rather than one wide draw.
    #
    # Everything used to be a single log-uniform over the full range, e.g.
    # trans over 0.3..250 px. That leaves coverage of any particular magnitude
    # to chance, and the bands that matter most were the thinnest: level 0 here
    # (sub-pixel camera motion, a near-tripod shot) did not exist at all, and
    # the object case started at 5 px so a slowly drifting subject was never
    # shown. Drawing the LEVEL first and the value inside it guarantees every
    # band is represented in proportion, which is what makes "progressively
    # larger" a property of the dataset rather than of the seed.
    #
    #        trans px        roll rad       sigma_flow px   obj_vel px      yaw/pitch rad
    LEVELS = (
        ((0.05, 0.5),   (0.0000, 0.002), (0.01, 0.04), (0.2, 1.0),    (0.0000, 0.0015)),
        ((0.5, 3.0),    (0.0000, 0.006), (0.04, 0.12), (1.0, 5.0),    (0.0000, 0.0040)),
        ((3.0, 15.0),   (0.0005, 0.015), (0.12, 0.40), (5.0, 25.0),   (0.0010, 0.0100)),
        ((15.0, 60.0),  (0.0010, 0.030), (0.40, 1.20), (25.0, 100.0), (0.0020, 0.0200)),
        ((60.0, 250.0), (0.0020, 0.055), (1.20, 6.00), (100.0, 400.0),(0.0040, 0.0350)),
    )

    def __init__(self, rng, regime=None, level=None):
        r = rng
        # 5 regimes now: 4 is parallax. See synth_burst.
        self.regime = regime if regime is not None else r.integers(0, 5)
        self.level = int(level) if level is not None else int(r.integers(0, len(self.LEVELS)))
        tr, th, sf, ov, yp = self.LEVELS[self.level]

        def logu(lo, hi):
            return float(np.exp(r.uniform(np.log(lo), np.log(hi))))

        self.trans = logu(*tr)
        # Up to 0.055 rad (3.2 deg) at the top level. Handheld roll across an
        # 8-frame burst reaches several degrees, and the mask was measured
        # FAILING on rotation -- 1.3 dB worse than merging everything on a
        # burst it had estimated correctly.
        self.theta = float(r.uniform(*th)) * (1 if r.random() < 0.7 else 0)
        # Irregular (non-monotonic) camera motion.
        #
        # Every trajectory used to be displacement = trans * n/(N-1), a straight
        # ramp whose only randomness was its direction. Real handheld is a
        # random walk: the camera reverses, pauses and accelerates inside one
        # burst, so consecutive frames are not ordered by how far they have
        # moved. That matters to the mask because span, |E| and the residual
        # are all read per frame against frame 0, and a ramp makes frame index
        # a near-perfect proxy for misalignment -- a shortcut that does not
        # exist on a real burst.
        self.irregular = bool(r.random() < 0.5)
        self.sigma_flow = logu(*sf)
        # A translation-only estimate of a rotating field: coherent within-tile
        # error, the mode a residual test cannot see.
        self.translation_only = bool(self.regime == 1)
        if self.translation_only:
            self.theta = max(self.theta, float(r.uniform(0.002, max(0.003, th[1]))))
            self.sigma_flow = min(self.sigma_flow, 0.8)
        # An independently moving subject, now from 0.2 px (a slow drift the
        # block match follows exactly) up to 400 px (which it cannot follow at
        # all). The mask has to handle both ends and everything between.
        self.has_object = bool(self.regime == 2)
        self.obj_vel = logu(*ov)
        # Parallax: depth-induced motion under camera TRANSLATION.
        #
        # Different in kind from the moving-object case, which is why it gets
        # its own regime rather than a wider obj_vel. An object is one compact
        # support moving against a static background; parallax is EVERYWHERE,
        # its magnitude is tied to scene structure rather than to a mask, and
        # at every depth discontinuity a tile straddles two disparities -- so
        # the single per-tile motion vector lands between them and is wrong on
        # both sides. That is the same geometry as the side-of-frame
        # misalignment that survives rotation, and no residual test sees it,
        # because within a tile the error is coherent.
        #
        # disparity = baseline / depth, so the inverse-depth map IS the flow
        # shape and only its scale is free.
        # Out-of-plane rotation. Drawn independently per axis and allowed to be
        # zero, because a handheld burst is not always tilting.
        # Dolly, log-symmetric about 1.0 so in and out are equally likely.
        dz = (yp[1] - yp[0]) * 1.2
        self.scale = float(np.exp(r.uniform(-dz, dz))) if r.random() < 0.5 else 1.0
        self.yaw = float(r.uniform(*yp)) * (1 if r.random() < 0.6 else 0)
        self.pitch = float(r.uniform(*yp)) * (1 if r.random() < 0.6 else 0)
        self.has_parallax = bool(self.regime == 4)
        self.par_disp = logu(max(0.3, tr[0]), max(1.0, tr[1]))
        self.obj_lock_p = float(r.uniform(0.5, 1.0))
        # Non-rigid: the subject also rotates and changes scale, so its displaced
        # copy does not match itself the way a translated rectangle does.
        self.obj_theta = float(r.uniform(-0.06, 0.06))
        self.obj_scale = float(np.exp(r.uniform(np.log(0.94), np.log(1.06))))
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
    # A random walk when irregular, the old ramp otherwise. Normalised so the
    # walk's LARGEST excursion is spec.trans either way, which keeps the two
    # trajectory kinds comparable in magnitude rather than making the irregular
    # ones systematically smaller.
    if spec.irregular:
        stp = rng.standard_normal((N, 2))
        wlk = np.cumsum(stp, axis=0)
        wlk -= wlk[0]
        amp = np.abs(wlk).max()
        wlk = wlk / amp if amp > 1e-9 else wlk
        rot = np.cumsum(rng.standard_normal(N))
        rot -= rot[0]
        ramp = np.abs(rot).max()
        rot = rot / ramp if ramp > 1e-9 else rot
    for n in range(1, N):
        f = n / (N - 1.0)
        if spec.irregular:
            th = spec.theta * float(rot[n])
            tx = spec.trans * float(wlk[n, 0])
            ty = spec.trans * float(wlk[n, 1])
            yw = spec.yaw * float(wlk[n, 0])
            pt = spec.pitch * float(wlk[n, 1])
        else:
            th = spec.theta * f * rng.uniform(0.6, 1.4)
            tx = spec.trans * f * rng.uniform(-1, 1)
            ty = spec.trans * f * rng.uniform(-1, 1)
            yw = spec.yaw * f * rng.uniform(-1, 1)
            pt = spec.pitch * f * rng.uniform(-1, 1)
        sc_n = spec.scale ** (float(wlk[n, 0]) if spec.irregular else f)
        motions.append(Motion(th, tx, ty, cy, cx, yaw=yw, pitch=pt, scale=sc_n))
        if spec.has_object:
            ox = tx + spec.obj_vel * f * rng.uniform(-1, 1)
            oy = ty + spec.obj_vel * f * rng.uniform(-1, 1)
            # Its own rotation on top of the camera's, so the subject deforms
            # relative to the background rather than sliding rigidly.
            obj_motions.append(Motion(th + spec.obj_theta * f, ox, oy,
                                      cy, cx, yaw=yw, pitch=pt,
                                      scale=sc_n * spec.obj_scale ** f))
        else:
            obj_motions.append(motions[-1])

    # Object support in REFERENCE raw coordinates. A smoothed random field
    # thresholded to the wanted area, not a rectangle: a rectangle's displaced
    # copy still matches itself along its own edges, which is exactly the
    # evidence the mask is supposed to find.
    obj_mask = np.zeros((h, w), bool)
    if spec.has_object:
        frac = float(rng.uniform(0.10, 0.45))
        small = rng.standard_normal((max(2, h // 64), max(2, w // 64)))
        # box-blur the field a couple of times, then bilinearly enlarge
        for _ in range(2):
            small = 0.25 * (small +
                            np.roll(small, 1, 0) + np.roll(small, -1, 0) +
                            np.roll(small, 1, 1))
        yy_ = np.linspace(0, small.shape[0] - 1, h)
        xx_ = np.linspace(0, small.shape[1] - 1, w)
        y0i = np.clip(yy_.astype(int), 0, small.shape[0] - 2)
        x0i = np.clip(xx_.astype(int), 0, small.shape[1] - 2)
        fy_ = (yy_ - y0i)[:, None]
        fx_ = (xx_ - x0i)[None, :]
        top = small[y0i][:, x0i] * (1 - fx_) + small[y0i][:, x0i + 1] * fx_
        bot = small[y0i + 1][:, x0i] * (1 - fx_) + small[y0i + 1][:, x0i + 1] * fx_
        field = top * (1 - fy_) + bot * fy_
        obj_mask = field > np.quantile(field, 1.0 - frac)

    # Inverse depth, for the parallax regime. A smooth random field with a few
    # sharp steps: the smooth part gives the gradual disparity change a per-tile
    # flow handles fine, and the steps give the depth edges where it cannot.
    inv_depth = None
    if spec.has_parallax:
        k = max(2, min(h, w) // 96)
        f0 = rng.standard_normal((k, k))
        for _ in range(2):
            f0 = 0.25 * (f0 + np.roll(f0, 1, 0) + np.roll(f0, -1, 0) +
                         np.roll(f0, 1, 1))
        ys = np.linspace(0, k - 1, h)
        xs = np.linspace(0, k - 1, w)
        y0i = np.clip(ys.astype(int), 0, k - 2)
        x0i = np.clip(xs.astype(int), 0, k - 2)
        fy_ = (ys - y0i)[:, None]
        fx_ = (xs - x0i)[None, :]
        top = f0[y0i][:, x0i] * (1 - fx_) + f0[y0i][:, x0i + 1] * fx_
        bot = f0[y0i + 1][:, x0i] * (1 - fx_) + f0[y0i + 1][:, x0i + 1] * fx_
        fld = top * (1 - fy_) + bot * fy_
        fld = (fld - fld.min()) / max(fld.max() - fld.min(), 1e-9)
        # Two or three depth planes with hard edges, plus the smooth part.
        steps = float(rng.integers(2, 4))
        inv_depth = (np.floor(fld * steps) / steps) * 0.7 + fld * 0.3
        inv_depth = inv_depth.astype(np.float64)

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
        if inv_depth is not None:
            # Depth-proportional shift along the camera's own displacement for
            # this frame, so it is a true parallax field and not extra noise.
            # Motion stores the translation as .t -- an hasattr('tx') guard here
            # silently produced zero parallax.
            ux, uy = _par_dir(motions[n])
            px = px + ux * spec.par_disp * inv_depth
            py = py + uy * spec.par_disp * inv_depth
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
        if inv_depth is not None:
            # Per-PIXEL inverse depth: this is the exact field, which is what
            # makes the ground-truth merge correct and ferr meaningful.
            ux, uy = _par_dir(motions[n])
            idp = inv_depth[np.clip(np.rint(lr_y).astype(np.int64), 0, h - 1),
                            np.clip(np.rint(lr_x).astype(np.int64), 0, w - 1)]
            fx = fx + ux * spec.par_disp * idp
            fy = fy + uy * spec.par_disp * idp
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

    # Per-tile MEDIAN inverse depth: what a block match actually locks onto.
    #
    # It finds the one displacement that best explains the tile as a whole,
    # which is the dominant depth in it. A tile wholly inside one depth plane is
    # therefore estimated correctly; a tile straddling a depth EDGE gets a
    # single vector that is wrong on both sides of the edge, by half the
    # disparity step each way. That residual is coherent within the tile, so no
    # residual-based test can see it -- the same geometry as the side-of-frame
    # misalignment that survives rotation.
    tile_idp = None
    if inv_depth is not None:
        tile_idp = np.zeros((ny, nx))
        for i in range(ny):
            for j in range(nx):
                tile_idp[i, j] = np.median(
                    inv_depth[i * ts:(i + 1) * ts, j * ts:(j + 1) * ts])

    flows = []
    for n in range(N):
        bx, by = motions[n].forward(TCX, TCY)
        fx = bx - TCX
        fy = by - TCY
        if tile_idp is not None:
            ux, uy = _par_dir(motions[n])
            fx = fx + ux * spec.par_disp * tile_idp
            fy = fy + uy * spec.par_disp * tile_idp
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
                sigma_flow=spec.sigma_flow, h=h, w=w,
                # Returned so an evaluation can select the rotating bursts.
                # Rotation is estimated CORRECTLY here (the per-tile flow is
                # read off the true motion at tile centres) in every regime but
                # 1, so these are the bursts that answer "does the mask reject
                # camera rotation it could have merged".
                theta=spec.theta, trans=spec.trans,
                yaw=spec.yaw, pitch=spec.pitch, scale=spec.scale,
                irregular=spec.irregular, level=spec.level,
                has_parallax=spec.has_parallax,
                translation_only=spec.translation_only)
