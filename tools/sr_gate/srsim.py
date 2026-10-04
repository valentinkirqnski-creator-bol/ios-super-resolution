"""Faithful NumPy port of the SHIPPING robustness + merge math, plus a burst
synthesiser, for training the sr_gate network.

Why a port and not the C++ itself: the gate is trained against the MERGED
outcome, which needs a ground truth, which needs synthesised motion. Everything
here mirrors one specific configuration -- the one core/types.h ships:

    bayer_mode            = true          cfa = RGGB
    grey_method           = FFT           robustness_raw_resolution_enabled = true
      => robustness_fft_guide_active(): the guide is compute_grey_fft(raw),
         FULL raw resolution, ONE channel, LINEAR (not sqrt(raw)), and R
         therefore lands on the raw lattice.
    robustness_guide_sqrt_active() = false   (linear guide -> curve indexed by
                                              brightness directly)
    fft_guide_noise_energy = 0.25            (noise variance surviving the low pass)
    kernel = Steerable, selection = Linear, snr_auto_tune = true
    scale = 2, r_t = 0.12, r_s1 = 1.99, r_s2 = 12.0, r_Mt = 0.8
    merge_robustness_bilinear = true

Anything that differs from the C++ is called out with a "DEVIATION:" comment.

Source material is real Bayer DNG only (bursts/*/**.dng, hdrplus test payloads).
"""
from __future__ import annotations

import math
import numpy as np

# --------------------------------------------------------------------------
# configuration (the shipping values, core/types.h)
# --------------------------------------------------------------------------

ALPHA_DNG = 1.80710882e-4
BETA_DNG = 3.1937599182128e-6
FFT_GUIDE_NOISE_ENERGY = 0.25

R_T = 0.12
R_S1 = 1.99
R_S2 = 12.0
R_MT = 0.8
GEOM_REJECT_THRESHOLD = 0.0045
GEOM_NOISE_FLOOR_MULT = 2.0
GEOM_REJECT_THRESHOLD_RELATIVE = 0.02

SCALE = 2
N_BRIGHTNESS = 1000
K_STRETCH = 4.0
K_SHRINK = 2.0

# RGGB, Config::cfa default {{0,1},{1,2}}
CFA = np.array([[0, 1], [1, 2]], dtype=np.int32)


class Cfg:
    """The handful of Config fields this port reads."""

    def __init__(self, noise_gain=1.0, tile_size=16):
        # alpha/beta after the guide-quad weight and the WB gain. The simulator
        # works in an already-prewhitened, neutral-gain domain, so
        # noise_wb_gain() == 1 and noise_guide_weight() is the CFA count
        # reciprocal; noise_alpha() averages the three channels:
        #   (1/1 + 1/2 + 1/1)/3 = 5/6 for alpha, same weights for beta.
        w = (1.0 + 0.5 + 1.0) / 3.0
        self.noise_gain = noise_gain
        self.alpha = ALPHA_DNG * noise_gain * w
        self.beta = BETA_DNG * noise_gain * noise_gain * w
        # What the per-raw-pixel sensor noise actually is (before the guide
        # averaging that noise_guide_weight models), used by the synthesiser.
        self.alpha_sensor = ALPHA_DNG * noise_gain
        self.beta_sensor = BETA_DNG * noise_gain * noise_gain
        # Config::noise_alpha_robustness(): the FFT low pass keeps a quarter of
        # the white-noise variance.
        self.alpha_rob = self.alpha * FFT_GUIDE_NOISE_ENERGY
        self.beta_rob = self.beta * FFT_GUIDE_NOISE_ENERGY
        self.tile_size = tile_size
        # Which guide. nch == 1 is the full-resolution FFT guide (linear);
        # nch == 3 is the half-resolution decimated Bayer guide with the 1.4
        # sqrt transfer, which is what Metal builds and what the shipping config
        # selects with robustness_raw_resolution OFF.
        self.nch = 1
        self.guide_sqrt = False
        # Per-channel alpha/beta for the 3-channel guide: alpha_dng[c] *
        # wb_gain(c) * guide_weight(c), NOT divided by 3 and NOT scaled by the
        # FFT guide's 0.25. wb_gain is 1 here because the synthesiser works in a
        # neutral-gain domain; guide_weight is 1/count(c), so green gets 1/2.
        self.alpha_ch = [ALPHA_DNG * noise_gain * w
                         for w in (1.0, 0.5, 1.0)]
        self.beta_ch = [BETA_DNG * noise_gain * noise_gain * w
                        for w in (1.0, 0.5, 1.0)]
        # snr_auto_tune (core/snr_tuning.cpp tune_config_snr); filled by
        # tune_snr() once the reference frame exists.
        self.k_detail = 0.17
        self.k_denoise = 0.0
        self.D_th = 0.76
        self.D_tr = 1.12

    def use_decimated_guide(self):
        """The 3-channel half-resolution Bayer guide with the 1.4 sqrt transfer.

        Everything that depends on which guide is in use is set HERE, together,
        because getting one of them wrong is silent: the FFT guide's 0.25
        filtered-noise factor does not apply to this one, and the curves move to
        the sqrt domain and become per channel.
        """
        self.nch = 3
        self.guide_sqrt = True
        # noise_alpha_robustness() == noise_alpha() when the FFT guide is off.
        self.alpha_rob = self.alpha
        self.beta_rob = self.beta
        return self

    def tune_snr(self, ref_raw, std_curve):
        brightness = float(ref_raw.mean())
        sigma = curve_at(std_curve, brightness)
        snr = brightness / sigma if sigma > 1e-8 else 15.0
        snr = min(max(snr, 6.0), 30.0)

        def lerp(y0, y1):
            t = min(max((snr - 6.0) / 24.0, 0.0), 1.0)
            return y0 + (y1 - y0) * t

        self.k_detail = lerp(0.33, 0.25)
        self.k_denoise = lerp(5.0, 3.0)
        self.D_th = lerp(0.81, 0.71)
        self.D_tr = lerp(1.24, 1.0)
        # tune_config_snr also picks the alignment tile size from the same SNR:
        # Ts = 64/32/16 by band, capped at 32, and pipeline_paths passes
        # bm_tile_sizes[0] (== Ts) to compute_robustness.
        self.tile_size = 32 if snr <= 22.0 else 16
        return snr


# --------------------------------------------------------------------------
# noise curves -- fast_monte_carlo.unitary_MC, linear (non-sqrt) domain
# --------------------------------------------------------------------------

def get_non_linearity_bound(alpha, beta, tol=3.0):
    """robustness.cpp get_non_linearity_bound: the brightness range over which
    clipping at 0 and at 1 is more than `tol` sigma away, so the statistic is
    linear in sigma and does not need the Monte Carlo."""
    a, b, t = float(alpha), float(beta), float(tol)
    tol_sq = t * t
    xmin = tol_sq / 2.0 * (a + math.sqrt(tol_sq * a * a + 4.0 * b))
    inner = (2.0 + tol_sq * a) ** 2 - 4.0 * (1.0 + tol_sq * b)
    xmax = (2.0 + tol_sq * a - math.sqrt(max(0.0, inner))) / 2.0
    return xmin, xmax


def interp_mc_range(std_curve, diff_curve, alpha, beta, tol=3.0):
    """robustness.cpp interp_MC_range + prepare_noise_curve_spec.

    The pipeline runs the Monte Carlo ONLY at the two non-linear ends and fills
    everything between by interpolating the SQUARES of the endpoint values, then
    taking the root. Leaving this out was a real difference, not a rounding one:
    with raw MC on every bin the end-to-end mask differed from the pipeline's by
    1.9e-2 mean, and raising the patch count 10x did not move it at all, which is
    what showed the gap was systematic.
    """
    n = N_BRIGHTNESS
    xmin, xmax = get_non_linearity_bound(alpha, beta, tol)
    imin = int(math.ceil(xmin * n)) + 1
    imax = int(math.floor(xmax * n)) - 1
    if imin > n or imax <= imin:          # full MC, nothing to interpolate
        return std_curve, diff_curve
    imin = max(imin, 0)
    imax = min(imax, n)
    b0 = (imin - 1) / float(n)
    b1 = (imax + 1) / float(n)
    denom = b1 - b0
    if denom <= 0:
        return std_curve, diff_curve
    s2min, s2max = float(std_curve[imin]) ** 2, float(std_curve[imax]) ** 2
    d2min, d2max = float(diff_curve[imin]) ** 2, float(diff_curve[imax]) ** 2
    i = np.arange(imin, imax + 1)
    nb = (i / float(n) - b0) / denom
    std_curve[imin:imax + 1] = np.sqrt(np.maximum(
        nb * (s2max - s2min) + s2min, 0.0)).astype(np.float32)
    diff_curve[imin:imax + 1] = np.sqrt(np.maximum(
        nb * (d2max - d2min) + d2min, 0.0)).astype(np.float32)
    return std_curve, diff_curve


def build_noise_curves(alpha, beta, n_patches=20000, seed=1337):
    """std_curve / diff_curve over N_BRIGHTNESS+1 bins.

    Same estimator as unitary_MC: for each brightness b draw two independent
    3x3 patches of N(b, alpha*b+beta), clip to [0,1], take the population std
    of each and |mean1 - mean2|; average over patches.

    DEVIATION: numpy Generator rather than a seeded MT19937 matching the C++
    NumpyRandomState draw-for-draw, and fewer patches than 1e5. The curve is a
    smooth function of b and the MC error is a fraction of a percent, far below
    what the noise model itself does to the mask; verified against the closed
    form below.
    """
    rng = np.random.default_rng(seed)
    b = np.arange(N_BRIGHTNESS + 1, dtype=np.float64) / N_BRIGHTNESS
    s = np.sqrt(np.maximum(0.0, alpha * b + beta))
    std_curve = np.empty_like(b)
    diff_curve = np.empty_like(b)
    for i in range(b.size):
        g = rng.standard_normal((2, n_patches, 9))
        p = np.clip(b[i] + s[i] * g, 0.0, 1.0)
        m = p.mean(axis=2)
        sd = np.sqrt(((p - m[:, :, None]) ** 2).mean(axis=2))
        std_curve[i] = 0.5 * (sd[0] + sd[1]).mean()
        diff_curve[i] = np.abs(m[0] - m[1]).mean()
    return interp_mc_range(std_curve.astype(np.float32),
                           diff_curve.astype(np.float32), alpha, beta)


def build_noise_curves_sqrt(alpha, beta, n_patches=20000, seed=1337):
    """unitary_MC with sqrt_domain=True: 1.4's guide applies sqrt to the CLIPPED
    raw before the 3x3 patch statistics, so the curve has to as well. Stored at
    the LATENT brightness bin, which is why the mask indexes it by mean^2.

    No closed form for this one -- the sqrt is applied per sample, before the
    patch mean, so the statistic is not a scaling of the linear case.
    """
    rng = np.random.default_rng(seed)
    b = np.arange(N_BRIGHTNESS + 1, dtype=np.float64) / N_BRIGHTNESS
    sd = np.sqrt(np.maximum(0.0, alpha * b + beta))
    std_curve = np.empty_like(b)
    diff_curve = np.empty_like(b)
    for i in range(b.size):
        g = rng.standard_normal((2, n_patches, 9))
        pv = np.clip(b[i] + sd[i] * g, 0.0, 1.0)
        pv = np.sqrt(pv)
        m = pv.mean(axis=2)
        st = np.sqrt(((pv - m[:, :, None]) ** 2).mean(axis=2))
        std_curve[i] = 0.5 * (st[0] + st[1]).mean()
        diff_curve[i] = np.abs(m[0] - m[1]).mean()
    return interp_mc_range(std_curve.astype(np.float32),
                           diff_curve.astype(np.float32), alpha, beta)


def noise_curves_closed_form(alpha, beta):
    """The unclipped limit of build_noise_curves, exact away from 0 and 1.

    mean of 9 iid N(b, s^2): m1 - m2 ~ N(0, 2 s^2 / 9), so
        E|m1 - m2| = s * sqrt(2)/3 * sqrt(2/pi) = s * 2/(3 sqrt(pi))
    population std of 9 samples: sqrt(chi2_8)/3 * s, so
        E[std] = s * sqrt(2) * Gamma(4.5) / Gamma(4) / 3
    """
    b = np.arange(N_BRIGHTNESS + 1, dtype=np.float64) / N_BRIGHTNESS
    s = np.sqrt(np.maximum(0.0, alpha * b + beta))
    k_d = 2.0 / (3.0 * math.sqrt(math.pi))
    k_s = math.sqrt(2.0) * math.gamma(4.5) / math.gamma(4.0) / 3.0
    return (s * k_s).astype(np.float32), (s * k_d).astype(np.float32)


def curve_at(curve, brightness):
    """noise_curve_index: id = lround(1000 * b), clamped."""
    if not np.isfinite(brightness):
        return float(curve[0])
    i = int(np.round(1000.0 * brightness))
    return float(curve[min(max(i, 0), curve.size - 1)])


def curve_lookup(curve, brightness):
    """Vectorised curve_at."""
    b = np.where(np.isfinite(brightness), brightness, 0.0)
    i = np.rint(1000.0 * b).astype(np.int64)
    np.clip(i, 0, curve.size - 1, out=i)
    return curve[i]


# --------------------------------------------------------------------------
# guide + local statistics
# --------------------------------------------------------------------------

def compute_grey_fft(raw):
    """core/grey_pyramid.cpp compute_grey_fft: zero the outer half of the
    shifted spectrum on both axes (where the CFA checkerboard lives) and
    invert. One guide sample per RAW pixel, phase-correct, no decimation."""
    h, w = raw.shape
    f = np.fft.fftshift(np.fft.fft2(raw.astype(np.float64)))
    y0, x0 = h // 4, w // 4
    m = np.zeros((h, w), dtype=bool)
    m[y0:h - y0, x0:w - x0] = True
    f *= m
    return np.real(np.fft.ifft2(np.fft.ifftshift(f))).astype(np.float32)


def compute_guide_decimate3(raw, wb_undo=(1.0, 1.0, 1.0), curve=1):
    """core/robustness.cpp compute_guide, Bayer branch.

    One output pixel per 2x2 quad, per colour, divided by how many sites that
    colour has (green has two). wb_undo is wb[1]/wb[c], which takes the loader's
    prewhitening back out so the guide is 1.4's camera-native sqrt(raw); the
    synthesiser works in a neutral domain so it is (1,1,1) there. curve 1 is
    sqrt, which is what guide_curve defaults to in the app.
    """
    h, w = raw.shape
    g = np.zeros((h // 2, w // 2, 3), np.float32)
    # RGGB: (0,0)=R, (0,1)=G, (1,0)=G, (1,1)=B -- taken from CFA rather than
    # assumed, same as the C++ reads cfg.cfa.
    sums = np.zeros((h // 2, w // 2, 3), np.float64)
    cnt = np.zeros(3, np.float64)
    for i in range(2):
        for j in range(2):
            c = int(CFA[i, j])
            sums[..., c] += raw[i::2, j::2]
            cnt[c] += 1
    for c in range(3):
        v = sums[..., c] / max(cnt[c], 1.0) * wb_undo[c]
        g[..., c] = apply_guide_curve(v, curve)
    return g


def apply_guide_curve(v, curve):
    """Twin of apply_guide_curve in robustness.cpp. 1 = sqrt, 0 = linear."""
    if curve == 1:
        return np.sqrt(np.maximum(v, 0.0))
    if curve == 2:
        return np.clip(v, 0.0, 1.0) ** (1.0 / 2.2)
    if curve == 3:
        vc = np.clip(v, 0.0, 1.0)
        return np.where(vc <= 0.0031308, 12.92 * vc,
                        1.055 * vc ** (1.0 / 2.4) - 0.055)
    return v


def local_stats_3x3(g):
    """core/robustness.cpp local_stats_3x3: 3x3 box mean and population
    variance, edge-clamped."""
    h, w = g.shape[:2]
    pad = ((1, 1), (1, 1)) + ((0, 0),) * (g.ndim - 2)
    p = np.pad(g.astype(np.float64), pad, mode='edge')
    s = np.zeros(g.shape, dtype=np.float64)
    s2 = np.zeros(g.shape, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            v = p[i:i + h, j:j + w]
            s += v
            s2 += v * v
    m = s / 9.0
    var = s2 / 9.0 - m * m
    return m.astype(np.float32), var.astype(np.float32)


def guide_noise_var(cfg, brightness, ch=0, nch=1):
    """robustness.cpp guide_noise_var. Note it uses the AVERAGED
    noise_alpha_robustness(), not the per-channel pair -- the per-channel values
    are only for the Monte-Carlo curves -- and halves the green channel because
    the guide averaged two Bayer greens into it."""
    b = np.clip(np.nan_to_num(brightness, nan=0.0), 0.0, 1.0)
    v = np.maximum(cfg.alpha_rob * b + cfg.beta_rob, 0.0)
    if nch == 3 and ch == 1:
        v = v * 0.5
    return v


# --------------------------------------------------------------------------
# d^2 / sigma^2  (Eq. 6, single-channel guide)
# --------------------------------------------------------------------------

def sample_bilinear_or_inf(img, y, x):
    """sample_bilinear_or_inf: +inf outside [0, h) x [0, w)."""
    h, w = img.shape
    ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
    ys = np.where(ok, y, 0.0)
    xs = np.where(ok, x, 0.0)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = ys - y0
    fx = xs - x0
    top = img[y0, x0] + (img[y0, x1] - img[y0, x0]) * fx
    bot = img[y1, x0] + (img[y1, x1] - img[y1, x0]) * fx
    out = top + (bot - top) * fy
    return np.where(ok, out, np.inf)


def warp_sample_comp(plane, flow, cfg):
    """Sample a COMPARISON-frame plane where Eq. 6 fetches it: that pixel's tile
    flow taken nearest, then bilinear in space, +inf outside the frame. Used for
    the comparison frame's local variance, which Eq. 6 computes and discards."""
    h, w = plane.shape
    ty, tx = tile_index_grids(h, w, cfg.tile_size)
    fx = flow[..., 0][ty, tx]
    fy = flow[..., 1][ty, tx]
    yy, xx = np.mgrid[0:h, 0:w]
    return sample_bilinear_or_inf(plane, yy + fy, xx + fx)


def tile_index_grids(h, w, ts, nch=1):
    """patch_idy/patch_idx, exactly as compute_robustness_core indexes them.

    One channel means the guide is at RAW resolution, so the tile is y // ts.
    Three channels means it is half resolution, so the raw coordinate is
    2y + 0.5 and the tile is int((2y + 0.5) / ts) -- the extra half pixel is the
    quad centre, and dropping it shifts the whole tile grid by half a tile at the
    bottom of each tile.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    if nch == 3:
        return (((2.0 * yy + 0.5) / ts).astype(np.int64),
                ((2.0 * xx + 0.5) / ts).astype(np.int64))
    return yy // ts, xx // ts


def compute_d_sigma(ref_means, ref_vars, comp_means, flow, cfg, std_curve, diff_curve):
    """Eq. 6 for the one-channel guide: d_p is the warped difference of local
    means, then apply_noise_model's shrinkage / noise floor."""
    h, w = ref_means.shape[:2]
    nch = 1 if ref_means.ndim == 2 else ref_means.shape[2]
    ts = cfg.tile_size
    ty, tx = tile_index_grids(h, w, ts, nch)
    ty = np.clip(ty, 0, flow.shape[0] - 1)
    tx = np.clip(tx, 0, flow.shape[1] - 1)
    # The three-channel guide is half resolution, so the merge fetches at half
    # the raw displacement.
    fsc = 0.5 if nch == 3 else 1.0
    fx = flow[..., 0][ty, tx] * fsc
    fy = flow[..., 1][ty, tx] * fsc
    yy, xx = np.mgrid[0:h, 0:w]

    # Eq. 6 aggregates across channels FIRST and applies max()/shrinkage once,
    # which is not the same as summing per-channel max() -- see the comment in
    # apply_noise_model.
    sigma_ms_sq = np.zeros((h, w), np.float64)
    sigma_md_sq = np.zeros((h, w), np.float64)
    d_ms_sq = np.zeros((h, w), np.float64)
    d_md_sq = np.zeros((h, w), np.float64)
    # The largest PER-CHANNEL ratio, alongside the summed one.
    #
    # Summing first dilutes a misalignment that lives in one or two channels.
    # Measured on a white-on-red tile offset (probe_colour.py), the per-channel
    # maximum is 3.55x the summed ratio, and exp(-a) reads 0.159 summed against
    # 0.068 per-channel -- the summed form calls a plain colour-edge
    # misalignment less than half as much evidence as it is. Across a white/red
    # boundary the RED channel barely changes while green and blue change
    # completely, so two channels carry the signal and the denominator collects
    # variance from all three.
    a_max = np.zeros((h, w), np.float64)
    for ch in range(nch):
        rm = ref_means if nch == 1 else ref_means[..., ch]
        rv = ref_vars if nch == 1 else ref_vars[..., ch]
        cm = comp_means if nch == 1 else comp_means[..., ch]
        sc_ = std_curve if nch == 1 else std_curve[ch]
        dc_ = diff_curve if nch == 1 else diff_curve[ch]
        comp = sample_bilinear_or_inf(cm, yy + fy, xx + fx)
        d_p = np.abs(rm - comp)
        # sqrt guide: the curve is keyed by LATENT brightness and the guide mean
        # is sqrt(latent), so it is indexed by mean SQUARED.
        bidx = rm.astype(np.float64) ** 2 if cfg.guide_sqrt else rm
        sigma_ms_sq += rv.astype(np.float64)
        sigma_md_sq += curve_lookup(sc_, bidx).astype(np.float64) ** 2
        d_ms_sq += d_p.astype(np.float64) ** 2
        d_md_sq += curve_lookup(dc_, bidx).astype(np.float64) ** 2
        s_c = np.maximum(rv.astype(np.float64),
                         curve_lookup(sc_, bidx).astype(np.float64) ** 2)
        with np.errstate(invalid='ignore', divide='ignore'):
            a_c = d_p.astype(np.float64) ** 2 / np.maximum(s_c, 1e-20)
        a_max = np.maximum(a_max, np.nan_to_num(a_c, nan=0.0, posinf=1e30))

    sigma_sq = np.maximum(sigma_ms_sq, sigma_md_sq)
    with np.errstate(invalid='ignore', divide='ignore'):
        shrink = d_ms_sq / (d_ms_sq + d_md_sq)
    shrink = np.where(np.isfinite(shrink), shrink, 1.0)
    d_sq = d_ms_sq * shrink * shrink
    # The four terms, returned as well as their combination. Eq. 6 reduces them
    # with a max() and a shrinkage, and both of those are lossy: the ratio
    # d^2/sigma^2 cannot say WHICH term won the max, nor how much of the raw
    # residual the noise correction removed. build_features turns them into
    # channels 8-11.
    comps = dict(sigma_ms_sq=sigma_ms_sq, sigma_md_sq=sigma_md_sq,
                 d_ms_sq=d_ms_sq, d_md_sq=d_md_sq, shrink=shrink,
                 a_max=a_max)
    return d_sq.astype(np.float32), sigma_sq.astype(np.float32), comps


# --------------------------------------------------------------------------
# s (Eq. 7/8) and the analytic Wronski mask
# --------------------------------------------------------------------------

def _tile_span(flow):
    ny, nx = flow.shape[:2]
    p = np.pad(flow, ((1, 1), (1, 1), (0, 0)), mode='edge')
    # Edge padding duplicates the border tile, which is what the C++ bounds
    # check achieves for a min/max: out-of-range neighbours are skipped, so the
    # window only ever sees in-range tiles.
    stack = np.stack([p[i:i + ny, j:j + nx] for i in range(3) for j in range(3)],
                     axis=0)
    return stack.max(axis=0) - stack.min(axis=0)


def compute_s(flow, mt=R_MT, s1=R_S1, s2=R_S2):
    """compute_s: s1 where the 3x3 tile neighbourhood flow span exceeds Mt."""
    d = _tile_span(flow)
    irregular = (d[..., 0] ** 2 + d[..., 1] ** 2) > mt * mt
    return np.where(irregular, s1, s2).astype(np.float32), irregular


def flow_span(flow):
    """The same 3x3 span, as a magnitude in raw pixels (feature 4)."""
    d = _tile_span(flow)
    return np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2).astype(np.float32)


def geom_residual(flow, ts, h, w, nch=1):
    """E = (grad flow) . (offset from tile centre): the within-tile part of the
    motion a per-tile translation cannot represent (motion_geom_reject).
    Returns ex, ey at raw resolution."""
    ny, nx = flow.shape[:2]
    if ny < 3 or nx < 3:
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    ty, tx = tile_index_grids(h, w, ts, nch)
    ty = np.clip(ty, 0, ny - 1)
    tx = np.clip(tx, 0, nx - 1)

    def cl(a, hi):
        return np.clip(a, 0, hi - 1)

    inv2ts = 1.0 / (2.0 * ts)
    dxf, dyf = flow[..., 0], flow[..., 1]
    up, dn = cl(ty - 1, ny), cl(ty + 1, ny)
    lf, rt = cl(tx - 1, nx), cl(tx + 1, nx)
    gdxdx = (dxf[ty, rt] - dxf[ty, lf]) * inv2ts
    gdydx = (dyf[ty, rt] - dyf[ty, lf]) * inv2ts
    gdxdy = (dxf[dn, tx] - dxf[up, tx]) * inv2ts
    gdydy = (dyf[dn, tx] - dyf[up, tx]) * inv2ts
    yy, xx = np.mgrid[0:h, 0:w]
    # The offset from the tile centre is in RAW pixels, so a half-resolution
    # guide pixel has to be converted: rawx = 2x + 0.5.
    sc = 2.0 if nch == 3 else 1.0
    rawx = sc * xx + 0.5 * (sc - 1.0)
    rawy = sc * yy + 0.5 * (sc - 1.0)
    u = rawx - (tx + 0.5) * ts
    v = rawy - (ty + 0.5) * ts
    ex = gdxdx * u + gdxdy * v
    ey = gdydx * u + gdydy * v
    return ex.astype(np.float32), ey.astype(np.float32)


def guide_gradient(ref_means, nch=1):
    """The central difference compute_robustness_core takes on ref_means:
    CHANNEL 0 only, divided by the guide-to-raw scale so it is per raw pixel."""
    g = (ref_means if ref_means.ndim == 2 else ref_means[..., 0]).astype(np.float32)
    sc = 2.0 if nch == 3 else 1.0
    gx = np.empty_like(g)
    gy = np.empty_like(g)
    gx[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2])
    gx[:, 0] = 0.5 * (g[:, 1] - g[:, 0])
    gx[:, -1] = 0.5 * (g[:, -1] - g[:, -2])
    gy[1:-1, :] = 0.5 * (g[2:, :] - g[:-2, :])
    gy[0, :] = 0.5 * (g[1, :] - g[0, :])
    gy[-1, :] = 0.5 * (g[-1, :] - g[-2, :])
    return gx / sc, gy / sc


def local_min_5x5(r):
    """Eq. 9's 5x5 minimum, edge-clamped."""
    h, w = r.shape
    p = np.pad(r, 2, mode='edge')
    out = p[0:h, 0:w].copy()
    for i in range(5):
        for j in range(5):
            np.minimum(out, p[i:i + h, j:j + w], out=out)
    return out


def wronski_robustness(d_sq, sigma_sq, flow, ref_means, cfg, geom_reject=True):
    """compute_robustness_core for the shipping configuration: Eq. 6-9 with
    motion_geom_reject on (both the absolute and the relative criterion)."""
    h, w = d_sq.shape
    nch = 1 if ref_means.ndim == 2 else ref_means.shape[2]
    ts = cfg.tile_size
    ty, tx = tile_index_grids(h, w, ts, nch)
    ty = np.clip(ty, 0, flow.shape[0] - 1)
    tx = np.clip(tx, 0, flow.shape[1] - 1)
    S, _ = compute_s(flow, R_MT, R_S1, R_S2)
    s = S[ty, tx]

    with np.errstate(divide='ignore', invalid='ignore'):
        r = s * np.exp(-np.minimum(d_sq.astype(np.float64) /
                                   sigma_sq.astype(np.float64), 700.0))
    r = np.clip(np.nan_to_num(r, nan=0.0) - R_T, 0.0, 1.0)

    if geom_reject:
        ex, ey = geom_residual(flow, ts, h, w, nch)
        emag = np.sqrt(ex * ex + ey * ey)
        gx, gy = guide_gradient(ref_means, nch)
        gmag = np.sqrt(gx * gx + gy * gy)
        rej = (gmag * emag) > GEOM_REJECT_THRESHOLD
        sc = 2.0 if nch == 3 else 1.0
        bri = (ref_means if nch == 1 else ref_means.mean(axis=2))
        bri = np.clip(np.nan_to_num(bri, nan=0.0), 0.0, 1.0)
        nsig = np.sqrt(guide_noise_var(cfg, bri, 0, nch)) / sc
        gmag_dn = np.maximum(0.0, gmag - GEOM_NOISE_FLOOR_MULT * nsig)
        rej = rej | ((gmag_dn / (bri + 1e-4)) * emag >
                     GEOM_REJECT_THRESHOLD_RELATIVE)
        r = np.where(rej, 0.0, r)

    return local_min_5x5(r.astype(np.float32))


# --------------------------------------------------------------------------
# features for the gate
# --------------------------------------------------------------------------

NUM_FEATURES = 13
FEATURE_NAMES = ('exp_a', 'log_a', 'snr', 'subpix', 'span', 'emag', 'grad', 'dir_e',
                 'shrink', 'sigdom', 'd_rel', 'varmatch')

# Feature compressions. Every one lands in [0, 1] so the network sees a fixed
# scale independent of exposure, ISO and sensor. Kept as module constants
# because core/sr_gate.h has to use exactly these numbers.
F_LOG_A_SCALE = 1.0 / 8.0
F_SNR_SCALE = 1.0 / 8.0
F_SPAN_SCALE = 2.0
F_EMAG_SCALE = 0.5
F_GRAD_SCALE = 1.0 / 6.0
F_DIRE_SCALE = 1.0 / 6.0
F_SUBPIX_SCALE = 1.0 / 0.70710678
F_DREL_SCALE = 0.5
F_DREL_FLOOR = 0.01
F_VARMATCH_SCALE = 1.0 / 3.0


def build_features(d_sq, sigma_sq, ref_means, ref_vars, flow, cfg,
                   comps=None, comp_vars_warped=None):
    """The gate input. Eight per-pixel channels on the raw-resolution guide
    lattice, all in [0, 1].

      0 exp_a  exp(-d^2/sigma^2)                 Wronski exponential, unscaled
      1 log_a  log1p(d^2/sigma^2)/8              the tail exp() has saturated
      2 snr    log1p(signal var/noise var)/8     is there detail worth merging
      3 subpix |frac(flow)| / sqrt(1/2)          NEW PHASE: what SR feeds on
      4 span   3x3 tile flow span / ts * 2       local motion irregularity
      5 emag   |E| / 2                           within-tile translation error
      6 grad   log1p(|grad g| / sigma_n)/6       is there an edge to smear
      7 dir_e  log1p(|grad g . E| / sigma_n)/6   predicted error ACROSS the edge

    Channels 8-11 are the statistics Eq. 6 COLLAPSES -- and they are MEASURED
    NEUTRAL, so the shipped model does not use them. They are kept because the
    negative result is worth more than the code costs: it is the answer to "the
    mask only sees the ratio, surely the terms behind it carry more", which is a
    reasonable thing to expect and turns out to be false here.

    Two independent measurements, both on the same 40 held-out bursts:

      * matched training -- same data, same 4200 steps, same seed, 8 channels
        against 12 -- gives 44.07 dB either way. The first attempt looked like a
        0.07 dB LOSS, but that comparison was run to a wall-clock budget, and the
        wider input is slower per step, so the 12-channel net had silently had
        1900 fewer steps. Match steps, not minutes, when comparing feature sets.
      * a ridge probe against the ORACLE mask (probe_features.py) puts the
        incremental R^2 of channels 8-11 over 0-7 at -0.0027: no linear
        information beyond what the first eight already span. Per channel,
        shrink +0.0032, sigdom +0.0009, varmatch -0.0004, d_rel -0.0076.

    Why, most likely: sigma_ms^2 is ref_vars, which channel 2 already carries;
    shrink is a monotone function of d_ms^2/d_md^2, which moves with d^2/sigma^2;
    and d_rel is a rescaling of the same residual. The one genuinely new
    measurement was varmatch, the comparison frame's texture, and it adds nothing
    either. The reduction Eq. 6 performs turns out not to lose much.

    Costs avoided by not shipping them: 2049 parameters instead of 1761, a 50%
    wider feature plane on both backends, and four more compressions that
    core/sr_gate_shared.h and the Metal kernel would have to keep in lockstep.

    The description of each, for anyone re-testing them:

      8 shrink   d_ms^2/(d_ms^2 + d_md^2)      how much of the raw residual
                                               survived the noise correction. 0
                                               means the difference is entirely
                                               explainable as noise, 1 means it is
                                               far above the noise's own |mean
                                               difference| scale. The ratio
                                               conflates this with magnitude.
      9 sigdom   sigma_ms^2/(sigma_ms^2 + sigma_md^2)   WHICH term won the max()
                                               in Eq. 6: above 0.5 real local
                                               structure dominates, below it the
                                               noise floor does, i.e. a flat area
                                               where sigma^2 is a floor rather
                                               than a measurement.
     10 d_rel    log1p(d_ms/brightness)/2      the residual relative to CONTRAST
                                               rather than to noise. Exposure
                                               invariant, and it separates a big
                                               difference on a bright edge from
                                               the same difference in shadow,
                                               which the noise-relative ratio
                                               does not.
     11 varmatch |log((var_ref + n)/(var_comp + n))|/3   does the warped
                                               comparison frame have the same
                                               amount of texture here at all.
                                               Both backends already compute the
                                               comparison frame's local variance
                                               and throw it away
                                               ("byproduct, never read", Eq. 6
                                               only wants the means), so this
                                               channel is free. It sees occlusion
                                               and gross mismatch, which a
                                               difference of MEANS can miss when
                                               two different textures happen to
                                               average alike.

    3 is the channel that makes this different from a residual-only mask: a
    half-pixel offset produces a large d^2 and is exactly the sample placement
    the merge needs. 7 is the channel d^2 cannot see, because a coherent
    per-tile error produces a residual the noise model explains away.
    """
    h, w = d_sq.shape
    nch = 1 if ref_means.ndim == 2 else ref_means.shape[2]
    ts = cfg.tile_size
    ty, tx = tile_index_grids(h, w, ts, nch)
    ty = np.clip(ty, 0, flow.shape[0] - 1)
    tx = np.clip(tx, 0, flow.shape[1] - 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        a = d_sq.astype(np.float64) / sigma_sq.astype(np.float64)
    a = np.nan_to_num(a, nan=np.inf, posinf=np.inf)

    f = np.zeros((NUM_FEATURES, h, w), dtype=np.float32)
    f[0] = np.exp(-np.minimum(a, 60.0))
    f[1] = np.clip(np.log1p(np.minimum(a, 1e12)) * F_LOG_A_SCALE, 0.0, 1.0)

    # Summed across channels, each against its own floor -- the green channel's
    # is halved because the guide averaged two Bayer greens into it.
    var_sum = np.zeros((h, w), np.float64)
    nvar_sum = np.zeros((h, w), np.float64)
    bri_sum = np.zeros((h, w), np.float64)
    for ch in range(nch):
        rm = ref_means if nch == 1 else ref_means[..., ch]
        rv = ref_vars if nch == 1 else ref_vars[..., ch]
        b_cl = np.clip(np.nan_to_num(rm, nan=0.0), 0.0, 1.0)
        nvar_sum += guide_noise_var(cfg, b_cl, ch, nch)
        var_sum += np.maximum(rv.astype(np.float64), 0.0)
        bri_sum += b_cl
    nvar = nvar_sum
    sig_var = np.maximum(var_sum - nvar_sum, 0.0)
    f[2] = np.clip(np.log1p(sig_var / np.maximum(nvar_sum, 1e-20)) * F_SNR_SCALE,
                   0.0, 1.0)

    fx = flow[..., 0][ty, tx]
    fy = flow[..., 1][ty, tx]
    u = np.abs(fx - np.rint(fx))
    v = np.abs(fy - np.rint(fy))
    f[3] = np.clip(np.sqrt(u * u + v * v) * F_SUBPIX_SCALE, 0.0, 1.0)

    span = flow_span(flow)[ty, tx]
    f[4] = np.clip(span / ts * F_SPAN_SCALE, 0.0, 1.0)

    ex, ey = geom_residual(flow, ts, h, w, nch)
    emag = np.sqrt(ex * ex + ey * ey)
    f[5] = np.clip(emag * F_EMAG_SCALE, 0.0, 1.0)

    gx, gy = guide_gradient(ref_means, nch)
    gmag = np.sqrt(gx * gx + gy * gy)
    # Channel 0's own floor at the cross-channel mean brightness, in the same
    # per-raw-pixel units as the gradient above.
    sc = 2.0 if nch == 3 else 1.0
    bri = bri_sum / float(nch)
    nsig = np.sqrt(np.maximum(guide_noise_var(cfg, bri, 0, nch), 1e-20)) / sc
    f[6] = np.clip(np.log1p(gmag / nsig) * F_GRAD_SCALE, 0.0, 1.0)
    f[7] = np.clip(np.log1p(np.abs(gx * ex + gy * ey) / nsig) * F_DIRE_SCALE,
                   0.0, 1.0)

    if comps is None:
        return f
    # ---- 8: the largest PER-CHANNEL d^2/sigma^2 -------------------------
    # Shipped, so it must stay at index 8: the C++ consumes the first
    # SRG_FEATURES channels and the set has to remain a contiguous prefix.
    am = comps['a_max']
    f[8] = np.clip(np.log1p(np.minimum(am, 1e12)) * F_LOG_A_SCALE, 0.0, 1.0)
    # ---- 9-12: the terms Eq. 6 reduced away, measured neutral ------------
    f[9] = np.clip(comps['shrink'], 0.0, 1.0)
    sms = comps['sigma_ms_sq']
    smd = comps['sigma_md_sq']
    f[10] = np.clip(sms / np.maximum(sms + smd, 1e-30), 0.0, 1.0)
    d_ms = np.sqrt(np.maximum(comps['d_ms_sq'], 0.0))
    bri = np.clip(bri_sum / float(nch), 0.0, 1.0)
    # Denominator floored at 1% of full scale, not at an epsilon: with an
    # epsilon this channel correlates +0.34 with darkness alone, because any
    # residual is large next to a near-zero brightness, and it would be partly a
    # "this pixel is dark" detector. At 0.01 that drops to +0.27 with no change
    # in how much of the top end clips (1.5% either way, and those are genuine
    # large residuals rather than the dark-pixel artifact).
    f[11] = np.clip(np.log1p(np.minimum(d_ms / (bri + F_DREL_FLOOR), 1e12)) *
                    F_DREL_SCALE, 0.0, 1.0)
    if comp_vars_warped is None:
        f[12] = 0.0
    else:
        n = np.maximum(nvar, 1e-20)
        cv = comp_vars_warped.astype(np.float64)
        ok = np.isfinite(cv)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = (sms + n) / (np.where(ok, np.maximum(cv, 0.0), 0.0) + n)
            lr = np.abs(np.log(np.maximum(ratio, 1e-30)))
        # A fetch outside the comparison frame has no texture to compare with, so
        # it reads as maximal mismatch -- the same direction Eq. 6 takes it with
        # its +inf residual.
        f[12] = np.where(ok, np.clip(lr * F_VARMATCH_SCALE, 0.0, 1.0), 1.0)
    return f
