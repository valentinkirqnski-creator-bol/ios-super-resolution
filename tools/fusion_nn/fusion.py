"""Shared pieces of the fusion-weight prototype.

- burst I/O (the .npz written by gen.py)
- merge_terms: per frame and output pixel, the kernel-regression numerator A and denominator B
  of core/merge.cpp (accumulate_comp / accumulate_ref, core defaults), so that
      out_c = sum_f w_f A_fc / sum_f w_f B_fc
  is the app's merge for ANY per-frame weights w (R = analytic, softmax(logits) = the network)
- net_inputs: what the network sees per frame (variance-stabilised guide differences)
- FusionNet: the network (depthwise-separable, 12 channels)
- fuse / loss
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- burst file
class Burst:
    def __init__(self, path):
        z = np.load(path)
        self.path = path
        self.gain = z["gain"].astype(np.float32)
        self.frames = torch.from_numpy(z["codes"].astype(np.float32) / 4095.0 * self.gain[:, None, None])
        self.N, self.H, self.W = self.frames.shape
        self.covs = torch.from_numpy(z["covs"].astype(np.float32))          # [N, ch, cw, 3] xx xy yy
        self.flow = torch.from_numpy(z["flow"].astype(np.float32))          # [N-1, ny, nx, 2] raw px
        # analytic R (validation only in 2x data; training files assume a raw-res R, as the app's default)
        self.R = torch.from_numpy(z["R"].astype(np.float32) / 255.0) if "R" in z else None  # [N-1, rh, rw, rc]
        self.R_shape = tuple(self.R.shape[1:]) if self.R is not None else (self.H, self.W, 1)
        self.target = torch.from_numpy(z["target"].astype(np.float32))      # [H, W, 3]
        self.ghost = torch.from_numpy(z["ghost"].astype(np.float32) / 16.0)  # [N-1, H/2, W/2] px
        self.merged = torch.from_numpy(z["merged"].astype(np.float32)) if "merged" in z else None
        p = z["params"]
        self.a, self.b, self.tile = float(p[0]), float(p[1]), int(p[2])
        self.cfa = [[int(p[3]), int(p[4])], [int(p[5]), int(p[6])]]
        self.scale = float(p[7]) if len(p) > 7 else 1.0


# ---------------------------------------------------------------- the app's merge, per frame
def _tile_index(lr, rob_n, rob_is_raw, ts, nt):
    # merge.cpp seam_tile, nearest-robustness path (core defaults)
    if rob_is_raw:
        raw = torch.clamp(torch.floor(lr + 0.5), 0, rob_n - 1)
    else:
        g = torch.clamp(torch.floor((lr - 0.5) / 2 + 0.5), 0, rob_n - 1)
        raw = 2 * g + 0.5
    return torch.clamp(torch.floor(raw / ts), 0, nt - 1).long()


def _lround_pos(x):
    c = torch.floor(x)
    return (c + (x - c >= 0.5).float()).long()


def _interp_inv_cov(cov, ki, kj, comp):
    """merge.cpp interp_inv_cov, batched over frames. cov [F, ch, cw, 3]; ki, kj [F, h, w].
    comp: accumulate_comp's raw-det path (int() index); else accumulate_ref's."""
    F_, ch, cw = cov.shape[0], cov.shape[1], cov.shape[2]
    fx_ = kj - torch.trunc(kj)
    fy_ = ki - torch.trunc(ki)
    if comp:
        fx = torch.clamp(torch.trunc(kj), min=0).long()
        fy = torch.clamp(torch.trunc(ki), min=0).long()
    else:
        fx = torch.clamp(torch.floor(kj), min=0).long()
        fy = torch.clamp(torch.floor(ki), min=0).long()
    cx = torch.clamp(fx + 1, max=cw - 1)
    cy = torch.clamp(fy + 1, max=ch - 1)
    fx = torch.clamp(fx, max=cw - 1)
    fy = torch.clamp(fy, max=ch - 1)
    fb = torch.arange(F_)[:, None, None]
    tl, tr, bl, br = cov[fb, fy, fx], cov[fb, fy, cx], cov[fb, cy, fx], cov[fb, cy, cx]
    top = tl + fx_[..., None] * (tr - tl)
    bot = bl + fx_[..., None] * (br - bl)
    m = top + fy_[..., None] * (bot - top)
    xx, xy, yy = m[..., 0], m[..., 1], m[..., 2]
    det = xx * yy - xy * xy
    ok = det.abs() > 1e-10
    inv = 1.0 / torch.where(ok, det, torch.ones_like(det))
    ixx = torch.where(ok, inv * yy, torch.ones_like(det))
    ixy = torch.where(ok, -inv * xy, torch.zeros_like(det))
    iyy = torch.where(ok, inv * xx, torch.ones_like(det))
    mx = torch.maximum(ixx.abs(), torch.maximum(iyy.abs(), ixy.abs()))
    s = torch.where(mx > 32.0, 32.0 / mx, torch.ones_like(mx))
    return ixx * s, ixy * s, iyy * s


def _accumulate(frames, mx, my, valid, ixx, ixy, iyy, cfa, rloc=None):
    """3x3 taps around (mx, my) for every frame at once -> A, B [F, 3, h, w]."""
    F_, H, W = frames.shape
    h, w = mx.shape[1], mx.shape[2]
    A = torch.zeros(F_, 3, h * w)
    B = torch.zeros(F_, 3, h * w)
    imgf = frames.reshape(-1)
    base = (torch.arange(F_) * H * W)[:, None, None]
    cj = _lround_pos(mx.clamp(min=0))
    ci = _lround_pos(my.clamp(min=0))
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            j, i = cj + dj, ci + di
            ok = valid & (j >= 0) & (j < W) & (i >= 0) & (i < H)
            jc, ic = j.clamp(0, W - 1), i.clamp(0, H - 1)
            c = imgf[base + ic * W + jc]
            ch = cfa[ic & 1, jc & 1]
            dx = jc.float() - mx
            dy = ic.float() - my
            z = torch.clamp(ixx * dx * dx + 2 * ixy * dx * dy + iyy * dy * dy, min=0)
            wt = torch.exp(-0.5 * z) * ok.float()
            if rloc is not None:
                wt = wt * rloc.gather(-1, ch[..., None])[..., 0]
            idx = ch.view(F_, 1, -1)
            A.scatter_add_(1, idx, (wt * c).view(F_, 1, -1))
            B.scatter_add_(1, idx, wt.view(F_, 1, -1))
    return A.view(F_, 3, h, w), B.view(F_, 3, h, w)


def merge_terms(bu, y0, x0, h, w, R_analytic=False):
    """(output pixel coordinates; the raw position of output pixel h is h / bu.scale, as merge.cpp)"""
    """A, B [N, 3, h, w] for output window rows y0..y0+h, cols x0..x0+w (scale 1).
    Frame 0 is the reference (accumulate_ref); frames 1.. are accumulate_comp with R = 1
    (or with the analytic R when R_analytic, for the parity check / baseline)."""
    H, W = bu.H, bu.W
    yy, xx = torch.meshgrid(torch.arange(y0, y0 + h, dtype=torch.float32) / bu.scale,
                            torch.arange(x0, x0 + w, dtype=torch.float32) / bu.scale, indexing="ij")
    cfa = torch.tensor(bu.cfa, dtype=torch.long)
    rh, rw, rc = bu.R_shape
    rob_is_raw = (rh == H and rw == W)
    # reference
    mx0, my0 = xx[None], yy[None]
    ixx, ixy, iyy = _interp_inv_cov(bu.covs[0:1], (my0 - 0.5) / 2, (mx0 - 0.5) / 2, comp=False)
    A0, B0 = _accumulate(bu.frames[0:1], mx0, my0, torch.ones_like(mx0, dtype=torch.bool), ixx, ixy, iyy, cfa)
    # comparison frames
    fl = bu.flow
    px = _tile_index(xx, rw, rob_is_raw, bu.tile, fl.shape[2])
    py = _tile_index(yy, rh, rob_is_raw, bu.tile, fl.shape[1])
    v = fl[:, py, px]                                                           # [N-1, h, w, 2]
    mx, my = xx[None] + v[..., 0], yy[None] + v[..., 1]
    valid = (mx >= 0) & (mx < W) & (my >= 0) & (my < H)
    mxc, myc = mx.clamp(0, W - 1), my.clamp(0, H - 1)
    ixx, ixy, iyy = _interp_inv_cov(bu.covs[1:], (myc - 0.5) / 2, (mxc - 0.5) / 2, comp=True)
    rloc = None
    if R_analytic:
        ry, rx = (yy, xx) if rob_is_raw else ((yy - 0.5) / 2, (xx - 0.5) / 2)
        iy = torch.clamp(torch.floor(ry + 0.5), 0, rh - 1).long()
        ix = torch.clamp(torch.floor(rx + 0.5), 0, rw - 1).long()
        rloc = bu.R[:, iy, ix]                                                  # [N-1, h, w, rc]
        if rc == 1:
            rloc = rloc.expand(-1, -1, -1, 3)
    A1, B1 = _accumulate(bu.frames[1:], mx, my, valid, ixx, ixy, iyy, cfa, rloc)
    return torch.cat([A0, A1]), torch.cat([B0, B1])


def fuse(A, B, wts):
    """out [.., 3, h, w] = sum_f w_f A_f / sum_f w_f B_f ; A, B [.., N, 3, h, w], wts [.., N, 1|3, h, w]."""
    num = (wts * A).sum(-4)
    den = (wts * B).sum(-4)
    return num / den.clamp(min=1e-12)


# ---------------------------------------------------------------- network inputs
def guides(frames, cfa):
    """[N, H, W] raw -> [N, 3, H/2, W/2] quad RGB (R, mean G, B)."""
    out = torch.zeros(frames.shape[0], 3, frames.shape[1] // 2, frames.shape[2] // 2)
    cnt = torch.zeros(3)
    for dy in (0, 1):
        for dx in (0, 1):
            c = cfa[dy][dx]
            out[:, c] += frames[:, dy::2, dx::2]
            cnt[c] += 1
    return out / cnt[None, :, None, None]


def flow_at_guide(flow, gh, gw, tile):
    """Tile flow (raw px) at every guide pixel, the tile the merge reads there. [gh, gw, 2]"""
    g_y = torch.arange(gh, dtype=torch.float32)
    g_x = torch.arange(gw, dtype=torch.float32)
    ty = torch.clamp(torch.floor((2 * g_y + 0.5) / tile), 0, flow.shape[0] - 1).long()
    tx = torch.clamp(torch.floor((2 * g_x + 0.5) / tile), 0, flow.shape[1] - 1).long()
    return flow[ty[:, None], tx[None, :]]


def warp_guides(gd, bu):
    """Each comparison frame's guide sampled where the existing flow says the reference pixel is."""
    N, _, gh, gw = gd.shape
    out = gd.clone()
    yy, xx = torch.meshgrid(torch.arange(gh, dtype=torch.float32), torch.arange(gw, dtype=torch.float32), indexing="ij")
    for f in range(1, N):
        v = flow_at_guide(bu.flow[f - 1], gh, gw, bu.tile)
        sx = xx + v[..., 0] / 2
        sy = yy + v[..., 1] / 2
        grid = torch.stack([(sx + 0.5) / gw * 2 - 1, (sy + 0.5) / gh * 2 - 1], -1)[None]
        out[f] = F.grid_sample(gd[f:f + 1], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
    return out


def gat(x, a, b):
    return (2.0 / a) * torch.sqrt(torch.clamp(a * x + 0.375 * a * a + b, min=0))


N_IN = 10          # r1 models: 8 (no flow-residual channels)


def flow_residual(flow, gh, gw, tile):
    """Smoothly (bilinearly) interpolated tile flow minus the tile's own vector, at every guide
    pixel, raw px [gh, gw, 2]: where the single vector per tile does not fit (rotation, zoom,
    parallax), this is large -- the within-tile slip the merge will make."""
    ny, nx = flow.shape[0], flow.shape[1]
    def axis(n, nt):
        p = (2 * torch.arange(n, dtype=torch.float32) + 0.5) / tile - 0.5
        i0 = torch.clamp(torch.floor(p), 0, nt - 1).long()
        i1 = torch.clamp(i0 + 1, max=nt - 1)
        fr = torch.clamp(p - i0.float(), 0, 1)
        return i0, i1, fr
    y0, y1, fy = axis(gh, ny)
    x0, x1, fx = axis(gw, nx)
    top = flow[y0][:, x0] * (1 - fx)[None, :, None] + flow[y0][:, x1] * fx[None, :, None]
    bot = flow[y1][:, x0] * (1 - fx)[None, :, None] + flow[y1][:, x1] * fx[None, :, None]
    bil = top * (1 - fy)[:, None, None] + bot * fy[:, None, None]
    return bil - flow_at_guide(flow, gh, gw, tile)


def net_inputs(bu, n_in=N_IN):
    """[N, 8, H/2, W/2]: is-reference, reference brightness, (frame - reference) and
    (frame - burst mean), the last two in noise std (GAT) / 4, clamped to +-8."""
    gd = warp_guides(guides(bu.frames, bu.cfa), bu)
    g = gat(gd.clamp(min=0), bu.a, bu.b)
    ref = g[0:1]
    one = gat(torch.tensor(1.0), bu.a, bu.b)
    d_ref = ((g - ref) / 4).clamp(-8, 8)
    d_mean = ((g - g.mean(0, keepdim=True)) / 4).clamp(-8, 8)
    N, _, gh, gw = g.shape
    isref = torch.zeros(N, 1, gh, gw)
    isref[0] = 1
    lum = (ref.mean(1, keepdim=True) / one).expand(N, 1, gh, gw)
    x = torch.cat([isref, lum, d_ref, d_mean], 1)
    if n_in == 8:
        return x
    res = torch.zeros(N, 2, gh, gw)
    for f in range(1, N):
        res[f] = flow_residual(bu.flow[f - 1], gh, gw, bu.tile).permute(2, 0, 1).clamp(-4, 4)
    return torch.cat([x, res], 1)


# ---------------------------------------------------------------- network
class DSBlock(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=d, dilation=d, groups=c)
        self.pw = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x + F.relu(self.pw(self.dw(x)))


class FusionNet(nn.Module):
    """Per frame (shared weights) at guide resolution -> one fusion logit. Softmax over frames
    at each output pixel gives the frames' normalised contribution weights."""

    def __init__(self, c=12, dil=(1, 2, 4, 8), cin=N_IN):
        super().__init__()
        self.stem = nn.Conv2d(cin, c, 3, padding=1)
        self.blocks = nn.Sequential(*[DSBlock(c, d) for d in dil])
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)          # start: equal weights = merge with no robustness

    def forward(self, x):                        # [M, 8, gh, gw] -> [M, 1, gh, gw]
        return self.head(self.blocks(F.relu(self.stem(x))))


def weights_from_logits(logits, ref_cap=False):
    """logits [Bt, N, 1, gh, gw] -> normalised weights [Bt, N, 1, 2gh, 2gw] at output pixels
    (bilinear: output pixel x sits at guide coordinate (x - 0.5) / 2)."""
    Bt, N = logits.shape[:2]
    up = F.interpolate(logits.flatten(0, 1), scale_factor=2, mode="bilinear", align_corners=False)
    up = up.view(Bt, N, 1, *up.shape[-2:])
    if ref_cap:   # no comparison frame outweighs the reference (as R <= 1); on device: w_f/w_ref = exp(min(l_f - l_ref, 0))
        up = torch.cat([up[:, :1], torch.minimum(up[:, 1:], up[:, :1])], 1)
    return torch.softmax(up, 1)


# ---------------------------------------------------------------- loss
def noise_sigma(t, a, b):
    return torch.sqrt(a * t.clamp(min=0) + b)


def charb(x, eps=1e-2):
    return torch.sqrt(x * x + eps * eps)


def mis_weight(ghost, gamma):
    """Loss weight per output pixel from the true misalignment of the flow the merge reads
    (ghost [Bt, N-1, gh, gw] px): 1 where every frame is within 0.15 px, 1 + gamma from 0.5 px.
    Weights the loss only -- the network never sees it."""
    m = ((ghost.amax(1, keepdim=True) - 0.15) / 0.35).clamp(0, 1)
    return 1 + gamma * F.interpolate(m, scale_factor=2, mode="nearest")


def loss_fn(out, tgt, a, b, w_grad=0.5, wmap=None):
    """Charbonnier in noise std of one raw sample + the same on horizontal/vertical gradients
    (edge shifts), optionally weighted per pixel. out, tgt [Bt, 3, h, w]; a, b [Bt]; wmap [Bt, 1, h, w]."""
    s = noise_sigma(tgt, a[:, None, None, None], b[:, None, None, None])
    d = out - tgt
    if wmap is None:
        wmap = torch.ones_like(d[:, :1])
    wm = lambda x, w: (x * w).sum() / (w.sum() * x.shape[1])
    l0 = wm(charb(d / s), wmap)
    gx = (d[..., :, 1:] - d[..., :, :-1]) / (0.5 * (s[..., :, 1:] + s[..., :, :-1]))
    gy = (d[..., 1:, :] - d[..., :-1, :]) / (0.5 * (s[..., 1:, :] + s[..., :-1, :]))
    wx = torch.maximum(wmap[..., :, 1:], wmap[..., :, :-1])
    wy = torch.maximum(wmap[..., 1:, :], wmap[..., :-1, :])
    return l0 + w_grad * 0.5 * (wm(charb(gx), wx) + wm(charb(gy), wy))


# ---------------------------------------------------------------- DBSR-style network (2x)
N_XD = 10


def inputs_dbsr(bu):
    """gin [N, 3, G, G]: each frame's own (unwarped) guide, variance-stabilised, for the encoder;
    grid [N, G, G, 2]: where each guide pixel of the reference lies in each frame (the app's tile
    flow the merge uses), as grid_sample coordinates -- the encoder FEATURES are warped with it;
    xd [N, 10, G, G]: is-reference, (frame - reference) and (frame - burst mean) in noise std / 4
    (images warped by the same flow), flow residual (within-tile slip, raw px), noise level."""
    gd = guides(bu.frames, bu.cfa)
    N, _, gh, gw = gd.shape
    one = gat(torch.tensor(1.0), bu.a, bu.b)
    gin = gat(gd.clamp(min=0), bu.a, bu.b) * (16.0 / one)
    yy, xx = torch.meshgrid(torch.arange(gh, dtype=torch.float32), torch.arange(gw, dtype=torch.float32), indexing="ij")
    grid = torch.zeros(N, gh, gw, 2)
    grid[0] = torch.stack([(xx + 0.5) / gw * 2 - 1, (yy + 0.5) / gh * 2 - 1], -1)
    for f in range(1, N):
        v = flow_at_guide(bu.flow[f - 1], gh, gw, bu.tile)
        grid[f] = torch.stack([(xx + v[..., 0] / 2 + 0.5) / gw * 2 - 1, (yy + v[..., 1] / 2 + 0.5) / gh * 2 - 1], -1)
    g = gat(F.grid_sample(gd, grid, mode="bilinear", padding_mode="border", align_corners=False).clamp(min=0), bu.a, bu.b)
    d_ref = ((g - g[0:1]) / 4).clamp(-8, 8)
    d_mean = ((g - g.mean(0, keepdim=True)) / 4).clamp(-8, 8)
    isref = torch.zeros(N, 1, gh, gw)
    isref[0] = 1
    res = torch.zeros(N, 2, gh, gw)
    for f in range(1, N):
        res[f] = flow_residual(bu.flow[f - 1], gh, gw, bu.tile).permute(2, 0, 1).clamp(-4, 4)
    noise = torch.full((N, 1, gh, gw), float(torch.log(one)) / 5)
    return gin, grid, torch.cat([isref, d_ref, d_mean, res, noise], 1)


class FusionNetD(nn.Module):
    """DBSR-style (Bhat et al. 2021), sized for the phone: a shared encoder on every frame's own
    guide, the moving frame's features bilinearly warped by the app's flow, a decoder on
    [reference features, warped features, explicit differences] -> one fusion logit per frame
    and guide pixel. Softmax over frames (reference capped) = the frames' contribution weights."""

    def __init__(self, ce=12, c=16, dil=(1, 2, 4, 8)):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, ce, 3, padding=1), nn.ReLU(), DSBlock(ce, 1))
        self.mix = nn.Conv2d(2 * ce + N_XD, c, 1)
        self.blocks = nn.Sequential(*[DSBlock(c, d) for d in dil])
        self.head = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, gin, grid, xd):
        """gin [Bt, N, 3, Hg, Wg]; grid [Bt, N, h, w, 2] into gin; xd [Bt, N, 10, h, w] -> [Bt, N, 1, h, w]"""
        Bt, N = gin.shape[:2]
        f = self.enc(gin.flatten(0, 1))
        fw = F.grid_sample(f, grid.flatten(0, 1), mode="bilinear", padding_mode="border", align_corners=False)
        fw = fw.view(Bt, N, *fw.shape[1:])
        z = torch.cat([fw[:, :1].expand_as(fw), fw, xd], 2).flatten(0, 1)
        out = self.head(self.blocks(F.relu(self.mix(z))))
        return out.view(Bt, N, 1, *out.shape[-2:])

    def macs_per_px(self):
        ce, c = self.enc[0].out_channels, self.mix.out_channels
        enc = 3 * ce * 9 + ce * 9 + ce * ce
        dec = (2 * ce + N_XD) * c + len(self.blocks) * (c * 9 + c * c) + c
        return enc, dec


def weights_at(logits, gy, gx, ref_cap=True):
    """Fusion weights at output pixels whose guide coordinates (relative to the logits' window)
    are gy [Ho], gx [Wo]: bilinear logits (border clamp), reference cap, softmax over frames.
    logits [Bt, N, 1, h, w] -> [Bt, N, 1, Ho, Wo]."""
    Bt, N, _, h, w = logits.shape
    Y, X = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([(X + 0.5) / w * 2 - 1, (Y + 0.5) / h * 2 - 1], -1)[None].expand(Bt * N, -1, -1, -1)
    up = F.grid_sample(logits.flatten(0, 1), grid, mode="bilinear", padding_mode="border", align_corners=False)
    up = up.view(Bt, N, 1, *up.shape[-2:])
    if ref_cap:
        up = torch.cat([up[:, :1], torch.minimum(up[:, 1:], up[:, :1])], 1)
    return torch.softmax(up, 1)
