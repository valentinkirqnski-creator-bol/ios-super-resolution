"""Synthetic RAW burst formation with exact ground-truth correspondence.

World model: a stack of planar layers (background + foreground objects), each at
its own depth. For frame k and layer L we build a 3x3 homography
G[L,k]: frame-k raw pixel coords -> layer texture coords, composed of

    camera  : roll + zoom (walking toward) + perspective about a (possibly far,
              off-crop) optical centre, plus translation scaled by the layer's
              parallax factor (nearer layers move more, zoom differs by depth)
    object  : independent rigid/similarity motion of the layer (moving subjects)
    texture : world -> texel mapping

Because every pixel's colour comes from a known layer through a known
homography, the true correspondence ref->frame k and its visibility (occlusion,
disocclusion, out-of-frame) are exact, not estimated.

Image formation per frame: (sub-exposure average for motion blur) of the
optically pre-blurred textures -> per-channel white-balance gains -> exposure
(with per-frame jitter) -> RGGB mosaic -> Poisson-Gaussian noise -> clip at
white -> quantise.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

from . import scene
from .common import bayer_to_grey, gauss_blur


def _T(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float64)


def _R(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def _S(s):
    return np.diag([s, s, 1.0])


def _P(p1, p2):
    return np.array([[1, 0, 0], [0, 1, 0], [p1, p2, 1]], np.float64)


ZERO_CAM = dict(th=0.0, tx=0.0, ty=0.0, dz=0.0, p1=0.0, p2=0.0)
ZERO_OBJ = dict(mx=0.0, my=0.0, phi=0.0, sc=1.0)


class Layer:
    def __init__(self, rgba, origin, parallax, tex_scale, pad):
        # rgba: np [h,w,4]; pre-blurred later
        self.rgba = torch.from_numpy(np.ascontiguousarray(rgba.transpose(2, 0, 1))).float()
        self.origin = np.asarray(origin, np.float64)
        self.parallax = float(parallax)
        self.S = float(tex_scale)
        self.pad = pad
        _, self.Ht, self.Wt = self.rgba.shape

    def blur(self, sigma_texels):
        self.rgba = gauss_blur(self.rgba, sigma_texels)

    def frame_to_tex(self, cam, obj, center):
        s = 1.0 / (1.0 - cam["dz"] * self.parallax)
        C = _T(self.parallax * cam["tx"], self.parallax * cam["ty"]) @ _T(*center) \
            @ _P(cam["p1"], cam["p2"]) @ _R(cam["th"]) @ _S(s) @ _T(-center[0], -center[1])
        o = self.origin
        O = _T(obj["mx"], obj["my"]) @ _T(*o) @ _R(obj["phi"]) @ _S(obj["sc"]) @ _T(-o[0], -o[1])
        Tex = _T((self.Wt - 1) / 2.0, (self.Ht - 1) / 2.0) @ _S(self.S) @ _T(-o[0], -o[1])
        return Tex @ np.linalg.inv(O) @ np.linalg.inv(C)


def _apply_h(G, pts):
    """pts torch [P,2] float64 -> transformed [P,2] float64."""
    Gt = torch.from_numpy(G)
    X = pts @ Gt[:2, :2].T + Gt[:2, 2]
    w = pts @ Gt[2, :2] + Gt[2, 2]
    return X / w[:, None]


def _sample(layer, G, pts):
    t = _apply_h(G, pts)
    gx = t[:, 0] / (layer.Wt - 1) * 2 - 1
    gy = t[:, 1] / (layer.Ht - 1) * 2 - 1
    grid = torch.stack([gx, gy], 1).float().view(1, 1, -1, 2)
    out = F.grid_sample(layer.rgba[None], grid, mode="bilinear", padding_mode=layer.pad, align_corners=True)
    return out[0, :, 0, :]   # [4,P]


def _composite(layers, Gs, pts_per_layer):
    """Returns rgb [3,P] and per-layer contribution weights [L,P]."""
    P = pts_per_layer[0].shape[0]
    T = torch.ones(P)
    col = torch.zeros(3, P)
    w = [None] * len(layers)
    for i in reversed(range(len(layers))):           # nearest first
        s = _sample(layers[i], Gs[i], pts_per_layer[i])
        a = torch.ones(P) if i == 0 else s[3].clamp(0, 1)
        wi = T * a
        col += wi * s[:3]
        T = T * (1 - a)
        w[i] = wi
    return col, torch.stack(w)


def _interp(p, d, u):
    return {k: p[k] + u * d.get(k, 0.0) for k in p}


class Burst:
    """Holds a sampled world + motion and renders anything about it."""

    def __init__(self, sc, rng, Hr=256, Wr=256, tex_scale=2.0, scene_dir=None):
        self.sc, self.rng, self.Hr, self.Wr, self.S = sc, rng, Hr, Wr, tex_scale
        N = sc["n_frames"]
        self.N = N
        fc = np.array([(Wr - 1) / 2.0, (Hr - 1) / 2.0])
        ang = rng.uniform(0, 2 * np.pi)
        self.center = fc + sc["center_dist"] * np.array([math.cos(ang), math.sin(ang)])

        # ---- layers -------------------------------------------------------------------------
        M = 96
        Wt, Ht = int(tex_scale * (Wr + 2 * M)), int(tex_scale * (Hr + 2 * M))
        bg = scene.background(rng, Ht, Wt, sc["texture_mode"], scene_dir)
        if sc["highlights"]:
            bg = scene.add_highlights(rng, bg)
        self.layers = [Layer(np.concatenate([bg, np.ones_like(bg[..., :1])], 2), fc, 1.0, tex_scale, "reflection")]
        for _ in range(sc["n_objects"]):
            sz = rng.uniform(0.2, 0.6) * min(Wr, Hr)
            h = int(tex_scale * sz * rng.uniform(0.6, 1.4))
            w = int(tex_scale * sz * rng.uniform(0.6, 1.4))
            rgba = scene.object_rgba(rng, h, w, sc["texture_mode"])
            if sc["highlights"] and rng.random() < 0.5:
                rgba[..., :3] = scene.add_highlights(rng, rgba[..., :3], n=2)
            origin = np.array([rng.uniform(0.1, 0.9) * Wr, rng.uniform(0.1, 0.9) * Hr])
            par = rng.uniform(*sc["obj_parallax"])
            self.layers.append(Layer(rgba, origin, par, tex_scale, "zeros"))
        # nearer layers last (painter's order); background stays first
        self.layers[1:] = sorted(self.layers[1:], key=lambda l: l.parallax)
        so = sc["optics_sigma"] * tex_scale
        for l in self.layers:
            l.blur(so)

        # ---- motion -------------------------------------------------------------------------
        self.cams, self.objs, self.blur_cam, self.blur_obj = [], [], [], []
        drift = rng.normal(0, 1, 2)
        drift = drift / (np.linalg.norm(drift) + 1e-9) * sc["cam_drift"]
        rot_drift = sc["rot_drift"]          # radians per frame
        vel = []
        for _ in self.layers[1:]:
            a = rng.uniform(0, 2 * np.pi)
            v = sc["obj_speed"] * rng.uniform(0.3, 1.0)
            vel.append(dict(mx=v * math.cos(a), my=v * math.sin(a),
                            phi=math.radians(rng.normal(0, sc["obj_rot_deg"])),
                            sc=rng.normal(0, sc["obj_scale_rate"])))
        for k in range(N):
            if k == 0:
                cam = dict(ZERO_CAM)
                objs = [dict(ZERO_OBJ) for _ in self.layers]
            else:
                cam = dict(th=math.radians(rng.normal(0, sc["rot_deg"])) + rot_drift * k,
                           tx=rng.normal(0, sc["tremor"]) + drift[0] * k,
                           ty=rng.normal(0, sc["tremor"]) + drift[1] * k,
                           dz=sc["zoom_rate"] * k + rng.normal(0, sc["zoom_rate"] * 0.1),
                           p1=rng.normal(0, sc["persp"]), p2=rng.normal(0, sc["persp"]))
                objs = [dict(ZERO_OBJ)]
                for v in vel:
                    j = sc["obj_jitter"]
                    objs.append(dict(mx=v["mx"] * k + rng.normal(0, j), my=v["my"] * k + rng.normal(0, j),
                                     phi=v["phi"] * k, sc=1.0 + v["sc"] * k))
            self.cams.append(cam)
            self.objs.append(objs)
            blurred = rng.random() < sc["blur_prob"]
            b = sc["blur_px"] if blurred else 0.0
            a = rng.uniform(0, 2 * np.pi)
            self.blur_cam.append(dict(tx=b * math.cos(a), ty=b * math.sin(a),
                                      th=math.radians(rng.normal(0, 0.05)) if blurred else 0.0))
            # moving objects smear along their velocity by the exposure fraction
            ef = sc["exposure_frac"]
            self.blur_obj.append([dict()] + [dict(mx=v["mx"] * ef, my=v["my"] * ef) for v in vel])

        # ---- photometry ---------------------------------------------------------------------
        self.wb = np.array([1 / rng.uniform(1.5, 2.5), 1.0, 1 / rng.uniform(1.3, 2.2)], np.float32)
        self.E = [sc["exposure"] * (1 + (rng.uniform(-1, 1) * sc["exposure_jitter"] if k else 0.0))
                  for k in range(N)]
        self.a, self.b = sc["noise_a"], sc["noise_b"]
        self.bits = int(rng.choice([10, 12, 14]))

    # ------------------------------------------------------------------------------------------
    def mats(self, k, u=0.0):
        cam = _interp(self.cams[k], self.blur_cam[k], u)
        return [l.frame_to_tex(cam, _interp(self.objs[k][i], self.blur_obj[k][i], u), self.center)
                for i, l in enumerate(self.layers)]

    def subs(self, k):
        bc = self.blur_cam[k]
        moving = any(abs(d.get("mx", 0)) + abs(d.get("my", 0)) > 0.05 for d in self.blur_obj[k])
        if abs(bc["tx"]) + abs(bc["ty"]) + abs(bc["th"]) < 1e-9 and not moving:
            return [0.0]
        return list(np.linspace(-0.5, 0.5, 7))

    def render_rgb(self, k, pts):
        """Scene radiance seen by frame k at raw coords pts [P,2] (sub-exposure averaged)."""
        acc = None
        su = self.subs(k)
        for u in su:
            Gs = self.mats(k, u)
            c, _ = _composite(self.layers, Gs, [pts] * len(self.layers))
            acc = c if acc is None else acc + c
        return acc / len(su)

    def contributions(self, k, pts):
        Gs = self.mats(k, 0.0)
        return _composite(self.layers, Gs, [pts] * len(self.layers))[1]

    # ------------------------------------------------------------------------------------------
    def _pix(self, H, W, scale=1.0):
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64),
                                indexing="ij")
        x = (xx + 0.5) / scale - 0.5
        y = (yy + 0.5) / scale - 0.5
        return torch.stack([x.reshape(-1), y.reshape(-1)], 1)

    def _mosaic(self, rgb, k):
        Hr, Wr = self.Hr, self.Wr
        x = rgb.view(3, Hr, Wr) * torch.from_numpy(self.wb)[:, None, None] * self.E[k]
        raw = torch.empty(Hr, Wr)
        raw[0::2, 0::2] = x[0, 0::2, 0::2]
        raw[0::2, 1::2] = x[1, 0::2, 1::2]
        raw[1::2, 0::2] = x[1, 1::2, 0::2]
        raw[1::2, 1::2] = x[2, 1::2, 1::2]
        return raw

    def render(self, out_scale=1, seed=0):
        Hr, Wr, N = self.Hr, self.Wr, self.N
        gen = torch.Generator().manual_seed(int(seed))
        pts = self._pix(Hr, Wr)
        q = float(2 ** self.bits - 1)
        raws, cleans = [], []
        for k in range(N):
            raw = self._mosaic(self.render_rgb(k, pts), k)
            clean = raw.clamp(0, 1)
            var = (self.a * clean + self.b).clamp_min(0)
            noisy = (raw + var.sqrt() * torch.randn(raw.shape, generator=gen)).clamp(0, 1)
            noisy = torch.floor(noisy * q + 0.5) / q
            raws.append(noisy)
            cleans.append(clean)
        raws = torch.stack(raws)
        cleans = torch.stack(cleans)

        # ground truth: the reference view, noise-free, demosaicked, at the output grid
        Ho, Wo = Hr * out_scale, Wr * out_scale
        po = self._pix(Ho, Wo, out_scale)
        gt = self.render_rgb(0, po).view(3, Ho, Wo) * torch.from_numpy(self.wb)[:, None, None] * self.E[0]
        gt = gt.clamp(0, 1)

        # exact correspondence + visibility at grey-pixel centres
        Hg, Wg = Hr // 2, Wr // 2
        pg = self._pix(Hg, Wg)
        pg = pg * 2.0 + 0.5            # grey pixel centre in raw coords
        c0 = self.contributions(0, pg)
        Ls = c0.argmax(0)
        G0 = self.mats(0)
        flows = torch.zeros(N, 2, Hg, Wg)
        vis = torch.ones(N, Hg, Wg)
        for k in range(1, N):
            Gk = self.mats(k)
            qs = torch.stack([_apply_h(np.linalg.inv(Gk[i]) @ G0[i], pg) for i in range(len(self.layers))])
            q_ = qs[Ls, torch.arange(pg.shape[0])]
            ck = self.contributions(k, q_)
            v = ck[Ls, torch.arange(pg.shape[0])]
            inb = (q_[:, 0] >= 0) & (q_[:, 0] <= Wr - 1) & (q_[:, 1] >= 0) & (q_[:, 1] <= Hr - 1)
            vis[k] = (v * inb.float()).view(Hg, Wg).float()
            flows[k] = ((q_ - pg) / 2.0).T.reshape(2, Hg, Wg).float()

        return dict(raw=raws, raw_clean=cleans, grey=bayer_to_grey(raws), grey_clean=bayer_to_grey(cleans),
                    gt=gt, gt_flow=flows, vis=vis, layer_ref=Ls.view(Hg, Wg),
                    noise_a=float(self.a), noise_b=float(self.b), wb=self.wb, E=list(self.E))
