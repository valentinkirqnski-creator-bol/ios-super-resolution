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

    def __init__(self, sc, rng, Hr=256, Wr=256, tex_scale=2.0, scene_dir=None, bg_rgb=None,
                 cfa=((0, 1), (1, 2)), prewhite=None, allow_flip=True):
        """bg_rgb: optional camera-native linear RGB scene [h,w,3] (e.g. from a real DNG) used as the
        background instead of a procedural one; it is already sensor-space, so no extra WB is applied.
        cfa: 2x2 colour index per site (0=R,1=G,2=B). prewhite: per-colour gains applied to the noisy
        raw (how the app's loader delivers a pre-white-balanced DNG)."""
        self.cfa = cfa
        self.prewhite = None if prewhite is None else np.asarray(prewhite, np.float32)
        self.sc, self.rng, self.Hr, self.Wr, self.S = sc, rng, Hr, Wr, tex_scale
        N = sc["n_frames"]
        self.N = N
        fc = np.array([(Wr - 1) / 2.0, (Hr - 1) / 2.0])
        ang = rng.uniform(0, 2 * np.pi)
        self.center = fc + sc["center_dist"] * np.array([math.cos(ang), math.sin(ang)])

        # ---- layers -------------------------------------------------------------------------
        M = 96
        Wt, Ht = int(tex_scale * (Wr + 2 * M)), int(tex_scale * (Hr + 2 * M))
        if bg_rgb is not None:
            bh, bw = bg_rgb.shape[:2]
            if bh < Ht or bw < Wt:   # v38: a frame larger than the scene (full 12 MP at camera scale): mirror-pad
                py, px = max(0, (Ht - bh + 1) // 2), max(0, (Wt - bw + 1) // 2)
                bg_rgb = np.pad(bg_rgb, ((py, py), (px, px), (0, 0)), mode="reflect")
                bh, bw = bg_rgb.shape[:2]
            y0, x0 = int(rng.integers(0, bh - Ht + 1)), int(rng.integers(0, bw - Wt + 1))
            bg = np.ascontiguousarray(bg_rgb[y0:y0 + Ht, x0:x0 + Wt])
            fx_, fy_ = rng.random() < 0.5, rng.random() < 0.5
            if allow_flip and fx_:
                bg = bg[:, ::-1]
            if allow_flip and fy_:
                bg = bg[::-1]
            bg = np.ascontiguousarray(bg, dtype=np.float32)
        else:
            bg = scene.background(rng, Ht, Wt, sc["texture_mode"], scene_dir)
        if sc["highlights"]:
            bg = scene.add_highlights(rng, bg)
        self.layers = [Layer(np.concatenate([bg, np.ones_like(bg[..., :1])], 2), fc, 1.0, tex_scale, "reflection")]
        # Rotated tile content: square patches cut from the background itself (so the reference is
        # seamless) whose content rotates about the patch centre by a random angle in some comparison
        # frames. A one-vector-per-tile aligner cannot represent this: a true local-rotation misalignment.
        self.rot_patch_layers = []
        n_rand = int(sc.get("n_rot_patches", 0))
        n_edge = int(sc.get("n_rot_patches_edge", 0))
        if n_edge:   # edge/detail map of the background, sampled at candidate patch centres
            from scipy.ndimage import gaussian_filter
            gyy, gxx = np.gradient(bg.mean(-1))
            emap = gaussian_filter(gyy * gyy + gxx * gxx, 3.0 * tex_scale)
        for pi in range(n_rand + n_edge):
            if pi >= n_rand:                    # the EXTRA patches go on edges; the first n_rand stay random
                size = float(sc.get("rot_patch_tile", 32)) if rng.random() < 0.5 else float(rng.uniform(16, 64))
                cand = np.stack([rng.uniform(size / 2, Wr - size / 2, 64), rng.uniform(size / 2, Hr - size / 2, 64)], 1)
                tx_ = ((Wt - 1) / 2.0 + tex_scale * (cand[:, 0] - fc[0])).astype(int).clip(0, Wt - 1)
                ty_ = ((Ht - 1) / 2.0 + tex_scale * (cand[:, 1] - fc[1])).astype(int).clip(0, Ht - 1)
                w = emap[ty_, tx_] + 1e-12
                cx, cy = cand[rng.choice(64, p=w / w.sum())]
            elif rng.random() < 0.5:
                size = float(sc.get("rot_patch_tile", 32))                     # exactly one app tile
                cx = (int(rng.integers(0, Wr // int(size))) + 0.5) * size
                cy = (int(rng.integers(0, Hr // int(size))) + 0.5) * size
            else:
                size = float(rng.uniform(16, 64))
                cx, cy = rng.uniform(size / 2, Wr - size / 2), rng.uniform(size / 2, Hr - size / 2)
            half_tex = int(tex_scale * size * 0.75) + 2                        # room for the rotated square
            tcx = (Wt - 1) / 2.0 + tex_scale * (cx - fc[0])
            tcy = (Ht - 1) / 2.0 + tex_scale * (cy - fc[1])
            x0, y0 = int(round(tcx)) - half_tex, int(round(tcy)) - half_tex
            if x0 < 0 or y0 < 0 or x0 + 2 * half_tex + 1 > Wt or y0 + 2 * half_tex + 1 > Ht:
                continue
            crop = bg[y0:y0 + 2 * half_tex + 1, x0:x0 + 2 * half_tex + 1]
            yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]].astype(np.float32)
            hs = tex_scale * size / 2.0
            ox, oy = (tcx - x0), (tcy - y0)
            a = (np.clip(hs + 0.5 - np.abs(xx - ox), 0, 1) * np.clip(hs + 0.5 - np.abs(yy - oy), 0, 1))
            # texture centre must be the patch centre: shift by the sub-texel rounding
            origin = np.array([cx + (x0 + half_tex - tcx) / tex_scale, cy + (y0 + half_tex - tcy) / tex_scale])
            lay = Layer(np.concatenate([crop, a[..., None]], 2).astype(np.float32), origin, 1.0, tex_scale, "zeros")
            lay.rot_deg = [0.0] + [float(rng.choice([-1, 1]) * rng.uniform(0, sc.get("rot_max_deg", 33.0)))
                                   if rng.random() < 0.6 else 0.0 for _ in range(N - 1)]
            self.rot_patch_layers.append(lay)
        # v19: tile-sized patches centred on STRAIGHT edges (structure-tensor coherence x strength) whose content
        # tilts 0.3-4 deg AND shifts 0.05-0.6 px across the edge in some comparison frames: the one-vector-per-tile
        # aligner cannot follow, so a straight edge comes out tilted / offset inside one tile ("straight, then
        # shifted straight" at tile borders). No rng is drawn unless n_rot_patches_small > 0, so older bursts
        # rebuild bit-exactly.
        n_small = int(sc.get("n_rot_patches_small", 0))
        if n_small:
            from scipy.ndimage import gaussian_filter
            gyy, gxx = np.gradient(bg.mean(-1))
            jxx, jyy, jxy = (gaussian_filter(v, 2.0 * tex_scale) for v in (gxx * gxx, gyy * gyy, gxx * gyy))
            tr_, dd_ = jxx + jyy, np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
            l1_, l2_ = 0.5 * (tr_ + dd_), 0.5 * (tr_ - dd_)
            smap = ((l1_ - l2_) / (l1_ + l2_ + 1e-12)) ** 2 * l1_
        for _ in range(n_small):
            if sc.get("small_size"):     # v23: longer edge segments (draws no extra rng when unset)
                size = float(rng.uniform(*sc["small_size"]))
            else:
                size = float(sc.get("rot_patch_tile", 32)) if rng.random() < 0.5 else float(rng.uniform(16, 48))
            cand = np.stack([rng.uniform(size / 2, Wr - size / 2, 128), rng.uniform(size / 2, Hr - size / 2, 128)], 1)
            tx_ = ((Wt - 1) / 2.0 + tex_scale * (cand[:, 0] - fc[0])).astype(int).clip(0, Wt - 1)
            ty_ = ((Ht - 1) / 2.0 + tex_scale * (cand[:, 1] - fc[1])).astype(int).clip(0, Ht - 1)
            w = smap[ty_, tx_] ** 2 + 1e-30
            if sc.get("small_anywhere") and rng.random() < sc["small_anywhere"]:   # v28: some patches anywhere
                w = np.ones_like(w)
            ci = int(rng.choice(128, p=w / w.sum()))
            cx, cy = cand[ci]
            a_, b_, c_ = jxx[ty_[ci], tx_[ci]], jyy[ty_[ci], tx_[ci]], jxy[ty_[ci], tx_[ci]]
            l1c = 0.5 * (a_ + b_ + np.sqrt((a_ - b_) ** 2 + 4 * c_ * c_))
            nx_, ny_ = (l1c - b_, c_) if abs(l1c - b_) + abs(c_) > 1e-18 else (c_, l1c - a_)
            nn_ = math.hypot(nx_, ny_) + 1e-18
            nx_, ny_ = nx_ / nn_, ny_ / nn_                                     # edge normal (gradient direction)
            half_tex = int(tex_scale * size * 0.75) + 2
            tcx = (Wt - 1) / 2.0 + tex_scale * (cx - fc[0])
            tcy = (Ht - 1) / 2.0 + tex_scale * (cy - fc[1])
            x0, y0 = int(round(tcx)) - half_tex, int(round(tcy)) - half_tex
            if x0 < 0 or y0 < 0 or x0 + 2 * half_tex + 1 > Wt or y0 + 2 * half_tex + 1 > Ht:
                continue
            crop = bg[y0:y0 + 2 * half_tex + 1, x0:x0 + 2 * half_tex + 1]
            yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]].astype(np.float32)
            hs = tex_scale * size / 2.0
            ox, oy = (tcx - x0), (tcy - y0)
            a = (np.clip(hs + 0.5 - np.abs(xx - ox), 0, 1) * np.clip(hs + 0.5 - np.abs(yy - oy), 0, 1))
            origin = np.array([cx + (x0 + half_tex - tcx) / tex_scale, cy + (y0 + half_tex - tcy) / tex_scale])
            lay = Layer(np.concatenate([crop, a[..., None]], 2).astype(np.float32), origin, 1.0, tex_scale, "zeros")
            rot, sh = [0.0], [(0.0, 0.0)]
            # v22 (sc["small_mode"] = (tilt_lo, tilt_hi, shift_lo, shift_hi)): each affected frame gets ONE of
            # translation only / rotation only / both, at very small log-uniform magnitudes - the tiny slips that
            # thicken or double an edge. Unset: the v19 draws exactly as before.
            smode = sc.get("small_mode")
            # sc["small_consistent"]: ONE slip per patch, applied (with +-15% jitter) in ~90% of the frames - the
            # same region misaligned the same way across the burst, which the merge cannot average away.
            consistent = smode is not None and sc.get("small_consistent", False)
            if consistent:
                kind0 = int(rng.integers(0, 3)) if not sc.get("small_kinds") else int(rng.choice(sc["small_kinds"]))
                p_frame = float(rng.uniform(*sc["small_frac"])) if sc.get("small_frac") else 0.9
                tl, th, sl, sh_hi = smode
                r00 = float(rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(tl), np.log(th))))
                m00 = float(np.exp(rng.uniform(np.log(sl), np.log(sh_hi)))) * float(rng.choice([-1, 1]))
                tg00 = float(rng.normal(0, 0.3)) * m00
            dcons = None
            if sc.get("small_disp") and rng.random() < 0.5:   # v28: half the patches slip the SAME way in most frames
                dlo, dhi = sc["small_disp"]
                dcons = (int(rng.integers(0, 3)), float(rng.uniform(dlo, dhi)) * float(rng.choice([-1, 1])),
                         float(rng.choice([-1, 1]) * np.degrees(rng.uniform(dlo, dhi) / (size / 2.0))),
                         float(rng.normal(0, 0.5)), float(rng.uniform(0.4, 0.95)))
            for _ in range(N - 1):
                if dcons is not None:
                    kind, m, r_, tgf, pf = dcons
                    if rng.random() < pf:
                        jt = float(np.clip(1.0 + 0.15 * rng.normal(), 0.5, 1.5))
                        rot.append(r_ * jt if kind >= 1 else 0.0)
                        sh.append(((m * nx_ - tgf * m * ny_) * jt, (m * ny_ + tgf * m * nx_) * jt) if kind != 1
                                  else (0.0, 0.0))
                    else:
                        rot.append(0.0)
                        sh.append((0.0, 0.0))
                    continue
                if consistent:
                    if rng.random() < p_frame:
                        jt = float(np.clip(1.0 + 0.15 * rng.normal(), 0.5, 1.5))
                        rot.append(r00 * jt if kind0 >= 1 else 0.0)
                        sh.append(((m00 * nx_ - tg00 * ny_) * jt, (m00 * ny_ + tg00 * nx_) * jt) if kind0 != 1
                                  else (0.0, 0.0))
                    else:
                        rot.append(0.0)
                        sh.append((0.0, 0.0))
                    continue
                if sc.get("small_disp"):
                    # v28 (sc["small_disp"] = (lo, hi) raw px): each affected frame gets a translation, a rotation or
                    # both whose DISPLACEMENT is uniform in [lo, hi] - a rotation turns the patch so its border moves
                    # by that much (angle = disp / half size) - i.e. ghosts / blur / zipper edges of a known size
                    if rng.random() < sc.get("small_pframe", 0.7):
                        kind = int(rng.integers(0, 3))
                        dlo, dhi = sc["small_disp"]
                        m = float(rng.uniform(dlo, dhi)) * float(rng.choice([-1, 1]))
                        r_ = float(rng.choice([-1, 1]) * np.degrees(rng.uniform(dlo, dhi) / (size / 2.0)))
                        tg = float(rng.normal(0, 0.5)) * m
                        rot.append(r_ if kind >= 1 else 0.0)
                        sh.append((m * nx_ - tg * ny_, m * ny_ + tg * nx_) if kind != 1 else (0.0, 0.0))
                    else:
                        rot.append(0.0)
                        sh.append((0.0, 0.0))
                    continue
                if smode is not None:
                    if rng.random() < 0.7:
                        kind = int(rng.integers(0, 3))          # 0 translate, 1 rotate, 2 both
                        tl, th, sl, sh_hi = smode
                        r_ = float(rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(tl), np.log(th))))
                        m = float(np.exp(rng.uniform(np.log(sl), np.log(sh_hi)))) * float(rng.choice([-1, 1]))
                        tg = float(rng.normal(0, 0.3)) * m
                        rot.append(r_ if kind >= 1 else 0.0)
                        sh.append((m * nx_ - tg * ny_, m * ny_ + tg * nx_) if kind != 1 else (0.0, 0.0))
                    else:
                        rot.append(0.0)
                        sh.append((0.0, 0.0))
                    continue
                if rng.random() < 0.7:
                    rot.append(float(rng.choice([-1, 1]) * rng.uniform(0.3, 4.0)))
                    m = float(np.exp(rng.uniform(np.log(0.05), np.log(0.6)))) * float(rng.choice([-1, 1]))
                    tg = float(rng.normal(0, 0.2)) * m
                    sh.append((m * nx_ - tg * ny_, m * ny_ + tg * nx_))
                else:
                    rot.append(0.0)
                    sh.append((0.0, 0.0))
            lay.rot_deg, lay.shift = rot, sh
            self.rot_patch_layers.append(lay)
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
        # nearer layers last (painter's order); background stays first; rotated patches sit on the background
        self.layers[1:] = sorted(self.layers[1:], key=lambda l: l.parallax)
        self.layers[1:1] = self.rot_patch_layers
        so = sc["optics_sigma"] * tex_scale
        for l in self.layers:
            l.blur(so)

        # ---- motion -------------------------------------------------------------------------
        self.cams, self.objs, self.blur_cam, self.blur_obj = [], [], [], []
        drift = rng.normal(0, 1, 2)
        drift = drift / (np.linalg.norm(drift) + 1e-9) * sc["cam_drift"]
        rot_drift = sc["rot_drift"]          # radians per frame
        vel = []
        for l in self.layers[1:]:
            if hasattr(l, "rot_deg"):
                vel.append(None)
                continue
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
                if sc.get("motion_table") is not None:
                    # v37: replay a MEASURED camera motion (rows: rotation deg, shift x, shift y in px, zoom) frame
                    # by frame, scaled by sc["motion_gain"] (per-burst variety); no perspective, no tremor on top.
                    # The patch motions below are built exactly as without the table.
                    row = sc["motion_table"][min(k - 1, len(sc["motion_table"]) - 1)]
                    g = float(sc.get("motion_gain", 1.0))
                    cam = dict(th=math.radians(row[0] * g), tx=row[1] * g, ty=row[2] * g, dz=row[3] * g, p1=0.0, p2=0.0)
                else:
                    cam = dict(th=math.radians(rng.normal(0, sc["rot_deg"])) + rot_drift * k,
                               tx=rng.normal(0, sc["tremor"]) + drift[0] * k,
                               ty=rng.normal(0, sc["tremor"]) + drift[1] * k,
                               dz=sc["zoom_rate"] * k + rng.normal(0, sc["zoom_rate"] * 0.1),
                               p1=rng.normal(0, sc["persp"]), p2=rng.normal(0, sc["persp"]))
                objs = [dict(ZERO_OBJ)]
                for li, v in enumerate(vel):
                    if v is None:                                   # rotated-content patch
                        lay_ = self.layers[1 + li]
                        mx_, my_ = lay_.shift[k] if hasattr(lay_, "shift") else (0.0, 0.0)
                        objs.append(dict(mx=mx_, my=my_, phi=math.radians(lay_.rot_deg[k]), sc=1.0))
                        continue
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
            self.blur_obj.append([dict()] + [dict() if v is None else dict(mx=v["mx"] * ef, my=v["my"] * ef)
                                             for v in vel])

        # ---- photometry ---------------------------------------------------------------------
        self.wb = (np.ones(3, np.float32) if bg_rgb is not None else
                   np.array([1 / rng.uniform(1.5, 2.5), 1.0, 1 / rng.uniform(1.3, 2.2)], np.float32))
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
        for i in range(2):
            for j in range(2):
                raw[i::2, j::2] = x[self.cfa[i][j], i::2, j::2]
        return raw

    def _prewhiten(self, raw):
        if self.prewhite is None:
            return raw
        g = torch.empty_like(raw)
        for i in range(2):
            for j in range(2):
                g[i::2, j::2] = float(self.prewhite[self.cfa[i][j]])
        return raw * g

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
            raws.append(self._prewhiten(noisy))
            cleans.append(self._prewhiten(clean))
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
