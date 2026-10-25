"""Image formation with exact ground truth.

World (reused from tools/robust_nn/rnn/burst.py): planar layers -- the background scene, optional moving objects,
optional small "slipping" content patches -- each with an exact 3x3 homography per frame (camera roll / zoom /
perspective / translation about an optical centre, depth-scaled parallax, independent object motion).

Per frame k and sensor pixel p (raw coordinates: pixel centres at integers, x right, y down):

  1. radiance  L_k(x)       = composite of the optically blurred layer textures at the scene point frame k sees at x
                              (Gaussian PSF, sigma = optics_sigma sensor px; sub-exposure averaging for motion blur)
  2. aperture  E_k(p)       = mean of L_k over an n x n grid inside the pixel footprint p + [-fill/2, fill/2]^2
                              (box integration of the pixel area; aperture.n = 1 falls back to point sampling)
  3. exposure  s_k(p)       = E_k(p)[c(p)] * exposure_k            c(p) = the CFA colour at p
  4. noise     s~_k(p)      = s_k(p) + N(0, a * clip(s_k(p), 0, 1) + b)       (Poisson-Gaussian, DNG profile x ISO)
  5. clip / ADC              clip to [0, 1] (white level), quantise to 2^bits - 1 levels
  6. loader domain  r_k(p)  = s~ * prewhite[c(p)] * g_k           (the app's loader pre-white-balances; g_k is the
                              app's exposure matching gain, metadata ratio with a residual error, brightened pixels
                              capped at the site's saturation exactly like apply_raw_gain)

Ground truth recorded per frame: every layer homography, the camera / object parameters, the exposure, the gain.

Exact correspondence: for a reference pixel p, the layer the reference sees there (largest compositing weight) is
mapped through that layer's homographies into frame k: q_k(p). d_k(p) = q_k(p) - p is the ref->frame displacement
in raw px, the SAME direction as the app's flow (the merge samples frame k at p + flow). Visibility vis_k(p) = the
weight of that layer at q_k(p) in frame k (occlusion, disocclusion) x [q inside the frame].

HR target on the merge's output grid: output pixel o (integer, scale s) sits at reference raw coordinate o / s
(merge.cpp: lr_x = hr_j / scale, no half-pixel offset). T(o) = clip(exposure_0 * aperture-integrated radiance at o/s,
0, 1) * prewhite, all three channels: what the reference sensor would have measured with its own pixel aperture at
that point, i.e. the band-limited sensor signal sampled at twice the density. (The merge cannot undo the optics or
the aperture, so this is the reconstruction a perfect merge converges to.)

Simplifications (documented, configurable where it matters): global shutter (no rolling-shutter skew); Gaussian
PSF without chromatic aberration; box aperture with a uniform fill factor; no lens shading / vignetting; noise
uncorrelated between sites; colour = the source scene's own camera-native RGB (no spectral model); the procedural
moving objects are crops of the same scene with soft random silhouettes.
"""
import contextlib
import json
import math

import numpy as np
import torch

from rnn import burst as rb
from rnn import scene as rscene


@contextlib.contextmanager
def object_source(src, rng):
    """Moving objects cut from a real scene (soft random silhouette) instead of rnn.scene's procedural fills."""
    if src is None:
        yield
        return
    orig = rscene.object_rgba

    def obj(rng_, h, w, mode):
        H, W = src.shape[:2]
        h_, w_ = min(h, H - 2), min(w, W - 2)
        y0, x0 = int(rng_.integers(0, H - h_)), int(rng_.integers(0, W - w_))
        fill = np.ascontiguousarray(src[y0:y0 + h_, x0:x0 + w_])
        if (h_, w_) != (h, w):
            fill = np.pad(fill, ((0, h - h_), (0, w - w_), (0, 0)), mode="reflect")
        a = rscene._shape_mask(rng_, h, w)
        return np.concatenate([fill, a[..., None]], 2).astype(np.float32)

    rscene.object_rgba = obj
    try:
        yield
    finally:
        rscene.object_rgba = orig


class Formation:
    """One synthetic burst. Pixel coordinates are raw (sensor) px of an S x S frame."""

    def __init__(self, sc, rng, size, tex, scene_rgb, cfa, prewhite, aperture_n=3, fill=1.0, obj_src=None,
                 allow_flip=True):
        self.size, self.cfa = int(size), cfa
        self.prewhite = np.asarray(prewhite, np.float32)
        with object_source(obj_src, rng):
            self.b = rb.Burst(sc, rng, Hr=self.size, Wr=self.size, tex_scale=float(tex), bg_rgb=scene_rgb, cfa=cfa,
                              prewhite=None, allow_flip=allow_flip)
        self.N = self.b.N
        n = max(1, int(aperture_n))
        o = (np.arange(n) + 0.5) / n - 0.5
        self.offs = [(float(dx) * fill, float(dy) * fill) for dy in o for dx in o]
        self.gain = np.ones(self.N, np.float32)

    # ------------------------------------------------------------------------------------------ radiance
    def radiance(self, k, pts):
        """Aperture-integrated scene radiance (camera-native linear RGB, before exposure) seen by frame k for pixel
        centres pts [P,2] (float64 raw coords). -> [3,P] float32."""
        acc = None
        for dx, dy in self.offs:
            c = self.b.render_rgb(k, pts + torch.tensor([dx, dy], dtype=torch.float64))
            acc = c if acc is None else acc + c
        return acc / len(self.offs)

    def _grid(self, y0, x0, h, w, step=1.0):
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64),
                                indexing="ij")
        return torch.stack([(xx * step + x0).reshape(-1), (yy * step + y0).reshape(-1)], 1)

    # ------------------------------------------------------------------------------------------ frames
    def render(self, seed, gains=None, exposure_cap=True):
        """-> dict(raw [N,S,S] noisy loader-domain, clean [N,S,S] noise-free loader-domain, var [N,S,S] per-pixel
        noise variance in the loader domain, bits, exposure [N], gain [N]). gains: the exposure-matching gain per
        frame (None = 1)."""
        S, N = self.size, self.N
        gen = torch.Generator().manual_seed(int(seed))
        pts = self._grid(0, 0, S, S)
        q = float(2 ** self.b.bits - 1)
        wb = torch.from_numpy(self.b.wb)
        pw_site = torch.empty(S, S)
        col = torch.empty(S, S, dtype=torch.long)
        for i in range(2):
            for j in range(2):
                pw_site[i::2, j::2] = float(self.prewhite[self.cfa[i][j]])
                col[i::2, j::2] = self.cfa[i][j]
        g = np.ones(N, np.float32) if gains is None else np.asarray(gains, np.float32)
        self.gain = g
        raws, cleans, vars_ = [], [], []
        for k in range(N):
            rad = self.radiance(k, pts).view(3, S, S) * wb[:, None, None] * float(self.b.E[k])
            s = torch.gather(rad, 0, col[None]).squeeze(0)                 # CFA sampling
            clean = s.clamp(0, 1)
            var = (self.b.a * clean + self.b.b).clamp_min(0)
            noisy = (s + var.sqrt() * torch.randn(s.shape, generator=gen)).clamp(0, 1)
            noisy = torch.floor(noisy * q + 0.5) / q
            sat = clean >= 1.0
            # loader domain: prewhite x exposure-matching gain, brightened values capped at the site's saturation
            cap = pw_site
            def to_loader(x):
                p = x * pw_site
                qv = p * float(g[k])
                if exposure_cap and g[k] > 1.0:
                    qv = torch.where(qv > cap, torch.maximum(p, cap), qv)
                return qv
            raws.append(to_loader(noisy))
            cleans.append(to_loader(clean))
            v = var * (pw_site * float(g[k])) ** 2
            vars_.append(torch.where(sat, torch.zeros_like(v), v))
        return dict(raw=torch.stack(raws), clean=torch.stack(cleans), var=torch.stack(vars_), bits=self.b.bits,
                    exposure=np.asarray(self.b.E, np.float32), gain=g)

    # ------------------------------------------------------------------------------------------ ground truth
    def correspondence(self, pts):
        """Exact ref->frame correspondence for reference points pts [P,2] (raw coords).
        -> disp [N,2,P] (frame-k position minus reference position, raw px; frame 0 zeros), vis [N,P]."""
        b, S = self.b, self.size
        P = pts.shape[0]
        c0 = b.contributions(0, pts)
        Ls = c0.argmax(0)
        G0 = b.mats(0)
        disp = torch.zeros(self.N, 2, P, dtype=torch.float64)
        vis = torch.ones(self.N, P)
        idx = torch.arange(P)
        for k in range(1, self.N):
            Gk = b.mats(k)
            qs = torch.stack([rb._apply_h(np.linalg.inv(Gk[i]) @ G0[i], pts) for i in range(len(b.layers))])
            qk = qs[Ls, idx]
            ck = b.contributions(k, qk)
            v = ck[Ls, idx]
            inb = (qk[:, 0] >= 0) & (qk[:, 0] <= S - 1) & (qk[:, 1] >= 0) & (qk[:, 1] <= S - 1)
            vis[k] = (v * inb.float()).float()
            disp[k] = (qk - pts).T
        return disp, vis

    def correspondence_grid(self, y0, x0, h, w):
        """On the raw pixel grid [y0, y0+h) x [x0, x0+w). -> disp [N,2,h,w] float64, vis [N,h,w]."""
        d, v = self.correspondence(self._grid(y0, x0, h, w))
        return d.view(self.N, 2, h, w), v.view(self.N, h, w)

    def hr_target(self, oy0, ox0, h, w, scale):
        """T [3,h,w] (loader domain) for output pixels [oy0, oy0+h) x [ox0, ox0+w) at reference raw coords o/scale."""
        pts = self._grid(oy0 / scale, ox0 / scale, h, w, step=1.0 / scale)
        rad = self.radiance(0, pts).view(3, h, w) * torch.from_numpy(self.b.wb)[:, None, None] * float(self.b.E[0])
        pw = torch.from_numpy(self.prewhite)[:, None, None]
        return rad.clamp(0, 1) * pw

    def transforms(self):
        """Every frame's per-layer homography (frame raw px -> layer texture coords) and the motion parameters."""
        mats = np.stack([np.stack(self.b.mats(k)) for k in range(self.N)])          # [N, L, 3, 3]
        info = dict(cams=self.b.cams, center=[float(v) for v in self.b.center], exposure=[float(e) for e in self.b.E],
                    gain=[float(x) for x in self.gain], layers=len(self.b.layers),
                    rot_patches=len(self.b.rot_patch_layers), noise_a=float(self.b.a), noise_b=float(self.b.b),
                    bits=int(self.b.bits))
        return mats, json.dumps(info, default=float)


def rotation_h(deg, cx, cy, scale=1.0):
    """3x3 homography: rotation by deg (and isotropic scale) about (cx, cy), raw px."""
    t = math.radians(deg)
    c, s = math.cos(t) * scale, math.sin(t) * scale
    return np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0, 0, 1]], np.float64)
