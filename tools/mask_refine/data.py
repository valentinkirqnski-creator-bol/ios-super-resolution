"""Read the bursts gen_bursts.exe writes, and build the network's inputs.

The merge is linear in the mask, separately per frame, so the generator dumps
the per-frame factors rather than a finished picture:

    out = (A_ref + sum_n A_n * R_n) / (B_ref + sum_n B_n * R_n)

with A_n, B_n produced by the pipeline's own merge_comp() run at R == 1. Any
mask can therefore be applied here and the result is exactly what the pipeline
would have produced with it -- no reimplementation of the reconstruction, and
the gradient of the final image with respect to the mask is exact.

Mask polarity, taken from merge.cpp and asserted in test_invariant.py:
R = 1 accepts the sample, R = 0 removes it.
"""
from __future__ import annotations

import glob
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

HDR = '<8i f'


def read_burst(path):
    """-> dict of float32 arrays. Shapes:
         A_ref, B_ref, gt : (3, oh, ow)
         Rw, ferr         : (N, gh, gw)
         A, B             : (N, 3, oh, ow)
         flow             : (N, ny, nx, 2)
    """
    with open(path, 'rb') as f:
        n, gh, gw, oh, ow, ny, nx, ts = struct.unpack('<8i', f.read(32))
        gain = struct.unpack('<f', f.read(4))[0]
        rd = lambda k: np.frombuffer(f.read(4 * k), dtype='<f4')
        # The merge writes channel-interleaved (y, x, c); the network wants
        # (c, y, x), so every image is transposed once here rather than at
        # every use.
        chw = lambda a: np.ascontiguousarray(
            a.reshape(oh, ow, 3).transpose(2, 0, 1))
        A_ref = chw(rd(oh * ow * 3))
        B_ref = chw(rd(oh * ow * 3))
        gt = chw(rd(oh * ow * 3))
        Rw = np.zeros((n, gh, gw), np.float32)
        A = np.zeros((n, 3, oh, ow), np.float32)
        B = np.zeros((n, 3, oh, ow), np.float32)
        flow = np.zeros((n, ny, nx, 2), np.float32)
        ferr = np.zeros((n, gh, gw), np.float32)
        for i in range(n):
            Rw[i] = rd(gh * gw).reshape(gh, gw)
            A[i] = chw(rd(oh * ow * 3))
            B[i] = chw(rd(oh * ow * 3))
            flow[i] = rd(ny * nx * 2).reshape(ny, nx, 2)
            ferr[i] = rd(gh * gw).reshape(gh, gw)
    # compute_robustness_core (core/robustness.cpp:2187) writes r_val with no
    # isfinite guard, unlike the raw-resolution path 600 lines above which has
    # one. It emits a NaN about once in 1.3M pixels, and a NaN mask multiplies
    # straight through the merge into a NaN output pixel. Treated as fully
    # rejected here, which is what the guarded path does, so the training data
    # is not poisoned by a handful of them.
    nan = int((~np.isfinite(Rw)).sum())
    if nan:
        Rw = np.nan_to_num(Rw, nan=0.0, posinf=0.0, neginf=0.0)
    return dict(nan_mask_px=nan,
                n=n, gh=gh, gw=gw, oh=oh, ow=ow, ny=ny, nx=nx, ts=ts,
                gain=gain, A_ref=A_ref, B_ref=B_ref, gt=gt, Rw=Rw, A=A, B=B,
                flow=flow, ferr=ferr)


def merge(A_ref, B_ref, A, B, R, eps=1e-8):
    """The pipeline's merge, with R applied. Differentiable in R.

    A_ref,B_ref (b,3,H,W)   A,B (b,N,3,H,W)   R (b,N,gh,gw) -> (b,3,H,W)
    """
    Ro = upsample_mask(R, A.shape[-2], A.shape[-1]).unsqueeze(2)
    num = A_ref + (A * Ro).sum(1)
    den = B_ref + (B * Ro).sum(1)
    return num / den.clamp_min(eps)


def upsample_mask(R, oh, ow):
    """Mask lattice -> output lattice.

    merge.cpp samples the mask at the output pixel's own position mapped back
    through the guide scale, so nearest is the faithful choice: it reproduces
    "this output pixel belongs to that mask cell" exactly, where bilinear would
    invent a gradient the pipeline does not apply.
    """
    if R.shape[-2:] == (oh, ow):
        return R
    return F.interpolate(R, size=(oh, ow), mode='nearest')


# --------------------------------------------------------------------------
# network inputs
# --------------------------------------------------------------------------
# Eight channels, all on the mask lattice, all cheap. Listed here because the
# ablation in evaluate.py switches them off by name.
# The last four are WARPED-IMAGE evidence: the aligned frame itself and its
# signed difference from the reference, rather than a scalar summary of how
# different they are. The distinction matters because the failure being detected
# is one flow vector standing in for a varying field, and the trace that leaves
# is a signed +/- pair either side of an edge -- which |resid| erases, and which
# no derived statistic reconstructs.
FEATURES = ('mask', 'flow_mag', 'flow_span', 'resid', 'grad', 'luma',
            'frame_disagree', 'tile_phase',
            'warp_lum', 'resid_signed', 'resid_grad', 'resid_coarse')
NUM_FEATURES = len(FEATURES)


def _tile_grid(gh, gw, ny, nx):
    ty = np.clip((np.arange(gh) * ny // max(gh, 1)), 0, ny - 1)
    tx = np.clip((np.arange(gw) * nx // max(gw, 1)), 0, nx - 1)
    return ty, tx


def build_features(d, drop=()):
    """-> (N, 8, gh, gw) float32.

    Everything is derived from what the pipeline already produced: the mask, the
    flow field, and the merged frames. Nothing here needs the ground truth, so
    the same code runs at inference.
    """
    n, gh, gw = d['n'], d['gh'], d['gw']
    ny, nx = d['ny'], d['nx']
    out = np.zeros((n, NUM_FEATURES, gh, gw), np.float32)
    ty, tx = _tile_grid(gh, gw, ny, nx)

    # Per-frame merged luminance at mask resolution, for the residual and the
    # disagreement channels. A_n/B_n is frame n's own contribution, so A/B is
    # the image that frame alone would make.
    eps = 1e-6
    frames = []
    for i in range(n):
        fi = d['A'][i] / np.maximum(d['B'][i], eps)
        frames.append(_to_mask_lattice(fi.mean(0), gh, gw))
    ref = _to_mask_lattice(
        (d['A_ref'] / np.maximum(d['B_ref'], eps)).mean(0), gh, gw)
    stack = np.stack(frames, 0) if n else np.zeros((0, gh, gw), np.float32)

    gy, gx = np.gradient(ref)
    grad = np.sqrt(gx * gx + gy * gy)
    # Normalised by the frame's own scale so the network is not handed an
    # absolute brightness it would have to learn to discount.
    gnorm = grad / max(float(np.percentile(grad, 99)), 1e-5)
    lnorm = ref / max(float(np.percentile(ref, 99)), 1e-5)

    for i in range(n):
        fl = d['flow'][i]
        mag = np.sqrt((fl ** 2).sum(-1))[np.ix_(ty, tx)]
        # Span over the 3x3 tile neighbourhood: how badly a single vector
        # describes the local field, which is the parallax / motion-boundary
        # signature.
        pad = np.pad(fl, ((1, 1), (1, 1), (0, 0)), mode='edge')
        sp = np.zeros((ny, nx), np.float32)
        for a in range(3):
            for b in range(3):
                dd = pad[a:a + ny, b:b + nx] - fl
                sp = np.maximum(sp, np.sqrt((dd ** 2).sum(-1)))
        span = sp[np.ix_(ty, tx)]
        resid = np.abs(frames[i] - ref)
        if n > 1:
            others = np.delete(stack, i, axis=0)
            disagree = np.abs(frames[i][None] - others).mean(0)
        else:
            disagree = np.zeros_like(resid)
        ph = np.sqrt(((np.arange(gh)[:, None] * (ny / max(gh, 1)) % 1.0) - .5) ** 2 +
                     ((np.arange(gw)[None, :] * (nx / max(gw, 1)) % 1.0) - .5) ** 2)

        # --- warped-image evidence ---------------------------------------
        # frames[i] is frame i's own contribution through the real merge
        # kernels -- the warped frame as the pipeline actually fetched it.
        sgn = frames[i] - ref
        ss = max(float(np.percentile(np.abs(sgn), 99)), 1e-5)
        rgy, rgx = np.gradient(sgn)
        rg = np.sqrt(rgx * rgx + rgy * rgy)
        # Coarse residual: a box mean over ~1 alignment tile. A tile-wide
        # disagreement is the signature of the whole tile being fetched from the
        # wrong place, which is a different failure from a thin edge artefact
        # and wants a different decision.
        kb = max(4, gh // 64)
        pad = np.pad(sgn, kb, mode='edge')
        cs = np.cumsum(np.cumsum(pad, 0), 1)
        k2 = 2 * kb
        coarse = (cs[k2:, k2:] - cs[:-k2, k2:] - cs[k2:, :-k2] +
                  cs[:-k2, :-k2]) / float(k2 * k2)
        coarse = coarse[:gh, :gw]

        vals = {
            'mask': d['Rw'][i],
            'flow_mag': np.tanh(mag / 8.0),
            'flow_span': np.tanh(span / 4.0),
            'resid': np.tanh(resid / max(float(np.percentile(resid, 99)), 1e-5)),
            'grad': np.clip(gnorm, 0, 4),
            'luma': np.clip(lnorm, 0, 4),
            'frame_disagree': np.tanh(
                disagree / max(float(np.percentile(disagree, 99)), 1e-5)),
            'tile_phase': ph.astype(np.float32),
            'warp_lum': np.clip(frames[i] / max(float(np.percentile(ref, 99)),
                                                1e-5), 0, 4),
            # SIGNED, and tanh'd rather than clipped so both directions survive.
            'resid_signed': np.tanh(sgn / ss),
            'resid_grad': np.tanh(rg / max(float(np.percentile(rg, 99)), 1e-5)),
            'resid_coarse': np.tanh(coarse / ss),
        }
        for k, name in enumerate(FEATURES):
            out[i, k] = 0.0 if name in drop else vals[name]
    return out


def _to_mask_lattice(a, gh, gw):
    if a.shape == (gh, gw):
        return a.astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(a, np.float32))[None, None]
    return F.interpolate(t, size=(gh, gw), mode='area')[0, 0].numpy()


class Bursts:
    """Disk-backed burst store with a small in-memory cache.

    A 1024 px burst is ~145 MB, so holding sixty of them is 8 GB. They are read
    on demand instead, and the most recent `cache` kept, because a training step
    draws one burst and the next step usually wants a different one -- a few
    resident is enough to keep the disk out of the critical path without
    pretending the whole set fits.

    Training samples a WINDOW rather than the whole burst. The merge is pointwise
    in the output lattice once A and B are known, so a window is exactly what the
    pipeline would produce there; it keeps the step cheap enough to take many of
    them, and it is free augmentation. Evaluation uses the whole burst.
    """

    def __init__(self, root, drop=(), cache=6, window=0, seed=0):
        self.paths = sorted(glob.glob(os.path.join(root, 'burst_*.bin')))
        if not self.paths:
            raise SystemExit('no bursts in %s' % root)
        self.drop = tuple(drop)
        self.cache_n = max(1, cache)
        self.window = window
        self.rng = np.random.default_rng(seed)
        self._cache = {}
        self._order = []
        self.nan_mask_px = 0

    def __len__(self):
        return len(self.paths)

    def _get(self, i):
        if i in self._cache:
            return self._cache[i]
        d = read_burst(self.paths[i])
        d['feat'] = build_features(d, self.drop)
        self.nan_mask_px += d.get('nan_mask_px', 0)
        self._cache[i] = d
        self._order.append(i)
        while len(self._order) > self.cache_n:
            self._cache.pop(self._order.pop(0), None)
        return d

    @property
    def items(self):
        """Every burst, for evaluation. Loads them one at a time."""
        return _ItemView(self)

    def batch(self, i, device='cpu', window=None):
        d = self._get(i)
        win = self.window if window is None else window
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        if not win or win >= min(d['oh'], d['ow']):
            sl_o = (slice(None), slice(None))
            sl_m = (slice(None), slice(None))
        else:
            # Output window, and the mask window that governs it. The mask
            # lattice is a fixed integer fraction of the output, so the two
            # stay aligned.
            r = d['oh'] // d['gh']
            wm = max(8, win // r)
            my = int(self.rng.integers(0, max(1, d['gh'] - wm)))
            mx = int(self.rng.integers(0, max(1, d['gw'] - wm)))
            sl_m = (slice(my, my + wm), slice(mx, mx + wm))
            sl_o = (slice(my * r, (my + wm) * r), slice(mx * r, (mx + wm) * r))
        oy, ox = sl_o
        my_, mx_ = sl_m
        return (t(d['feat'][:, :, my_, mx_]).unsqueeze(0),
                t(d['Rw'][:, my_, mx_]).unsqueeze(0),
                t(d['A_ref'][:, oy, ox]).unsqueeze(0),
                t(d['B_ref'][:, oy, ox]).unsqueeze(0),
                t(d['A'][:, :, oy, ox]).unsqueeze(0),
                t(d['B'][:, :, oy, ox]).unsqueeze(0),
                t(d['gt'][:, oy, ox]).unsqueeze(0),
                t(d['ferr'][:, my_, mx_]).unsqueeze(0))


class _ItemView:
    def __init__(self, store):
        self.store = store

    def __getitem__(self, i):
        return self.store._get(i)

    def __len__(self):
        return len(self.store)
