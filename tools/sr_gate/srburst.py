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

def _par_dir(m):
    """Unit direction of a frame's translation. Parallax displaces along the
    camera's own motion, so a frame that did not move gets no parallax."""
    d = m.t
    n = float(np.hypot(d[0], d[1]))
    return (0.0, 0.0) if n < 1e-9 else (float(d[0]) / n, float(d[1]) / n)


# Half-res width of a whole 12 MP frame. par_disp is quoted for a frame this
# wide and rescaled to whatever is actually synthesised, so a crop behaves like
# a full frame instead of having most of its content displaced out of view.
REF_W = 2016.0


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

    def __init__(self, rng, regime=None):
        r = rng
        # 5 regimes: 4 is depth parallax. See synth_burst.
        self.regime = regime if regime is not None else r.integers(0, 5)
        # Handheld camera motion. Log-uniform because the small end is where
        # most shots live and where rejection is most harmful, while the large
        # end has to be represented at all.
        self.trans = float(np.exp(r.uniform(np.log(0.3), np.log(250.0))))
        self.theta = float(r.uniform(0.0, 0.020)) * (1 if r.random() < 0.7 else 0)
        self.sigma_flow = float(np.exp(r.uniform(np.log(0.08), np.log(6.0))))
        # ---- depth parallax (regime 4) -------------------------------------
        #
        # Measured on Downloads/burstsr, which is the motion this models: an
        # urban scene shot downward from height, sky to near street in one
        # frame. Global translation was 88-144 RAW px frame to frame, and after
        # removing it the residual was p50 135-146 raw px, p90 >250, max ~385.
        # Three measurements say that residual is DEPTH and not noise or
        # rotation: a global affine fit did not reduce it at all (135.0 ->
        # 136.3), it is largest where the image is most textured (flat windows
        # 100 px, textured 211 px -- the near field is the detailed part), and
        # it grows toward the bottom of the frame (124 -> 168 px, corr +0.29),
        # which is the near end when the camera points down.
        #
        # Frames here are HALF resolution, so those raw figures halve.
        #
        # par_disp is ABSOLUTE and not scaled to the crop, unlike a global
        # translation would be: parallax size is set by the scene's DEPTH RANGE,
        # not by how wide a field the crop covers. A 768 px crop holding a
        # roofline against sky carries the same disparity step as the whole
        # frame does. The range below is calibrated so the per-pixel flow error
        # at a depth edge brackets the 135-146 px median measured on burstsr.
        self.has_parallax = bool(self.regime == 4)
        self.par_disp = float(np.exp(r.uniform(np.log(12.0), np.log(150.0))))
        # Depth planes. A real street scene is not a smooth depth ramp: it has
        # hard steps at every building edge and roofline, and those steps are
        # where one vector per tile cannot be right on both sides.
        self.par_steps = int(r.integers(2, 5))
        # A translation-only estimate of a rotating field: coherent within-tile
        # error, the mode a residual test cannot see.
        self.translation_only = bool(self.regime == 1)
        if self.translation_only:
            self.theta = float(r.uniform(0.002, 0.020))
            self.sigma_flow = float(np.exp(r.uniform(np.log(0.08), np.log(0.8))))
        # An independently moving subject. 5-400 px log-uniform: block matching
        # follows the low end and cannot follow the high end, and the mask has to
        # handle both.
        self.has_object = bool(self.regime == 2)
        self.obj_vel = float(np.exp(r.uniform(np.log(5.0), np.log(400.0))))
        self.obj_lock_p = float(r.uniform(0.5, 1.0))
        # Non-rigid: the subject also rotates and changes scale, so its displaced
        # copy does not match itself the way a translated rectangle does.
        self.obj_theta = float(r.uniform(-0.06, 0.06))
        self.obj_scale = float(np.exp(r.uniform(np.log(0.94), np.log(1.06))))
        self.outlier_p = float(r.uniform(0.0, 0.06)) if self.regime == 3 else 0.0
        self.noise_gain = float(np.exp(r.uniform(np.log(1.0), np.log(24.0))))
        self.n_frames = 8


def structured_flow_error(ny, nx, sigma, rng):
    """Alignment error shaped the way parallax and camera rotation shape it.

    Independent Gaussian noise per tile -- what this used to inject -- is the one
    kind of error that does NOT occur in practice. A real block match goes wrong
    for reasons that are smooth or piecewise smooth across the frame:

      * ROTATION, yaw, pitch, dolly: the true displacement varies continuously
        across the frame, so a per-tile constant lags it by an amount that drifts
        slowly and coherently;
      * PARALLAX: displacement is constant within a depth plane and STEPS at
        every depth edge, so the error is near zero over most of a region and
        jumps hard at object boundaries.

    Both are coherent INSIDE a tile, which is what makes them invisible to any
    test built on the flow field's own derivatives, and both are correlated
    BETWEEN neighbouring tiles, which independent noise destroys.

    So: a smooth low-frequency field, plus a few hard steps, per component.
    """
    def smooth(k):
        f = rng.standard_normal((max(2, ny // k + 2), max(2, nx // k + 2)))
        yi = np.clip((np.arange(ny) / k).astype(int), 0, f.shape[0] - 2)
        xi = np.clip((np.arange(nx) / k).astype(int), 0, f.shape[1] - 2)
        fy = (np.arange(ny) / k - yi)[:, None]
        fx_ = (np.arange(nx) / k - xi)[None, :]
        a = f[yi][:, xi] * (1 - fx_) + f[yi][:, xi + 1] * fx_
        b = f[yi + 1][:, xi] * (1 - fx_) + f[yi + 1][:, xi + 1] * fx_
        return a * (1 - fy) + b * fy

    out = []
    for _ in range(2):
        g = smooth(max(2, int(rng.integers(3, 9))))
        # Hard steps: a few random half-planes, which is what a depth edge or an
        # occlusion boundary does to the matched displacement.
        st = np.zeros((ny, nx))
        for _ in range(int(rng.integers(0, 4))):
            yy, xx = np.mgrid[0:ny, 0:nx]
            th = rng.uniform(0, np.pi)
            c = (xx - nx / 2) * np.cos(th) + (yy - ny / 2) * np.sin(th)
            st += (c > rng.uniform(-nx / 3, nx / 3)) * rng.normal(0, 1.0)
        v = 0.65 * g + 0.35 * st
        sd = float(v.std())
        out.append(v / sd * sigma if sd > 1e-9 else v)
    return out[0], out[1]


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
            # Its own rotation on top of the camera's, so the subject deforms
            # relative to the background rather than sliding rigidly.
            obj_motions.append(Motion(th + spec.obj_theta * f, ox, oy, cy, cx))
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

    # Inverse depth for the parallax regime: 1/depth IS the shape of the
    # disparity field, so only its scale is free. A smooth random field with a
    # few hard steps, plus a vertical ramp, because a camera pointed downward
    # has its near field at the bottom -- which is the +0.29 row/residual
    # correlation measured on burstsr.
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
        ramp = np.linspace(0.0, 1.0, h)[:, None] * np.ones((1, w))
        fld = 0.55 * fld + 0.45 * ramp
        st = float(spec.par_steps)
        inv_depth = ((np.floor(fld * st) / st) * 0.72 + fld * 0.28).astype(np.float64)

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
            ux, uy = _par_dir(motions[n])
            pd = spec.par_disp
            px = px + ux * pd * inv_depth
            py = py + uy * pd * inv_depth
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
            ux, uy = _par_dir(motions[n])
            pd = spec.par_disp
            idp = inv_depth[np.clip(np.rint(lr_y).astype(np.int64), 0, h - 1),
                            np.clip(np.rint(lr_x).astype(np.int64), 0, w - 1)]
            fx = fx + ux * pd * idp
            fy = fy + uy * pd * idp
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

    # Per-tile MEDIAN inverse depth: what a block match locks onto.
    #
    # It finds the single displacement that best explains the tile as a whole,
    # which is its dominant depth. A tile wholly inside one depth plane is then
    # estimated correctly, but a tile straddling a depth EDGE -- a roofline
    # against sky, a car against road -- gets one vector that is wrong on BOTH
    # sides of the edge. That residual is coherent across the tile, so no
    # residual-based test can see it from the flow field, and it is the failure
    # this regime exists to teach.
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
            pd = spec.par_disp
            fx = fx + ux * pd * tile_idp
            fy = fy + uy * pd * tile_idp
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
            # Structured, not independent per tile. See structured_flow_error:
            # the errors parallax and rotation actually produce are smooth or
            # piecewise smooth across the frame, and that correlation between
            # neighbouring tiles is exactly what the mask has spatial context to
            # exploit. A third of bursts keep the old independent draw so the
            # uncorrelated case is still represented.
            if rng.random() < 0.67:
                ex, ey = structured_flow_error(ny, nx, spec.sigma_flow, rng)
            else:
                ex = rng.standard_normal((ny, nx)) * spec.sigma_flow
                ey = rng.standard_normal((ny, nx)) * spec.sigma_flow
            fx = fx + ex
            fy = fy + ey
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
