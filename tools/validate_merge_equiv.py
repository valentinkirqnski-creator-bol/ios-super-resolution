#!/usr/bin/env python3
"""
Host-side equivalence check: CPU merge.cpp semantics vs Metal HHSRKernels.metal
merge_accumulate_comp / merge_accumulate_ref (literal transcription).

Run: python tools/validate_merge_equiv.py
Exit 0 iff max|cpu-metal| == 0 on all synthetic cases (same float32 ops / std::exp).
"""
from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def f32(x: float) -> float:
    return float(np.float32(x))


def trunc_f(x: float) -> float:
    return f32(math.trunc(f32(x)))


def floor_f(x: float) -> float:
    return f32(math.floor(f32(x)))


def fabs_f(x: float) -> float:
    return f32(abs(f32(x)))


def isfinite_f(x: float) -> bool:
    return math.isfinite(f32(x))


def exp_f(x: float) -> float:
    # C++ std::exp / metal::precise::exp ΓÇö same IEEE formula; bit-match via numpy float32
    return f32(np.exp(np.float32(x)))


def round_half_away(x: float) -> int:
    """C++ lround / Metal round for x >= 0: half away from zero."""
    x = f32(x)
    return int(f32(np.floor(np.float32(x) + np.float32(0.5))))


def soften_inv_cov(ixx: float, ixy: float, iyy: float) -> Tuple[float, float, float]:
    k_max_abs = f32(32.0)
    m = max(fabs_f(ixx), max(fabs_f(iyy), fabs_f(ixy)))
    if not (m > k_max_abs) or not isfinite_f(m):
        if not isfinite_f(ixx) or not isfinite_f(ixy) or not isfinite_f(iyy):
            return f32(2.0), f32(0.0), f32(2.0)
        return ixx, ixy, iyy
    s = f32(k_max_abs / m)
    return f32(ixx * s), f32(ixy * s), f32(iyy * s)


def invert_sym_2x2(xx: float, xy: float, yy: float) -> Tuple[float, float, float]:
    det = f32(xx * yy - xy * xy)
    if fabs_f(det) > f32(1e-10):
        det_i = f32(1.0 / det)
        return f32(yy * det_i), f32(-xy * det_i), f32(xx * det_i)
    return f32(1.0), f32(0.0), f32(1.0)


def interp_inv_cov(
    covs: np.ndarray, kmap_i: float, kmap_j: float, raw_det: bool
) -> Tuple[float, float, float]:
    """covs: [h,w,4]"""
    h, w, _ = covs.shape
    frac_x = f32(kmap_j - trunc_f(kmap_j))
    frac_y = f32(kmap_i - trunc_f(kmap_i))
    if raw_det:
        fx = max(int(f32(kmap_j)), 0)  # toward-zero cast like C++ (int)
        fy = max(int(f32(kmap_i)), 0)
    else:
        fx = max(int(floor_f(kmap_j)), 0)
        fy = max(int(floor_f(kmap_i)), 0)
    cx = min(fx + 1, w - 1)
    cy = min(fy + 1, h - 1)

    def lerp2(idx: int) -> float:
        tl = f32(covs[fy, fx, idx])
        tr = f32(covs[fy, cx, idx])
        bl = f32(covs[cy, fx, idx])
        br = f32(covs[cy, cx, idx])
        top = f32(tl + frac_x * (tr - tl))
        bot = f32(bl + frac_x * (br - bl))
        return f32(top + frac_y * (bot - top))

    xx, xy, yy = lerp2(0), lerp2(1), lerp2(3)
    if raw_det:
        det = f32(xx * yy - xy * xy)
        if fabs_f(det) > f32(1e-10):
            inv_det = f32(1.0 / det)
            ixx = f32(inv_det * yy)
            ixy = f32(-inv_det * xy)
            iyy = f32(inv_det * xx)
        else:
            ixx, ixy, iyy = f32(1.0), f32(0.0), f32(1.0)
    else:
        ixx, ixy, iyy = invert_sym_2x2(xx, xy, yy)
    return soften_inv_cov(ixx, ixy, iyy)


@dataclass
class Cfg:
    scale: float = 1.0
    bayer: bool = True
    iso: bool = False
    tile_size: int = 16
    nch: int = 3
    cfa: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 1), (1, 2))
    robustness_denoise: bool = False
    rad_max: int = 2
    max_multiplier: float = 8.0
    max_frame_count: float = 2.0


def cfa_ch(cfg: Cfg, i: int, j: int) -> int:
    if not cfg.bayer:
        return 0
    return cfg.cfa[i & 1][j & 1]


def accumulate_comp_cpu(
    img, flow, covs, rob, num, den, y0: int, cfg: Cfg
) -> None:
    """Literal merge.cpp accumulate_comp."""
    band_h, Ws = num.shape[0], num.shape[1]
    lr_h, lr_w = img.shape
    for local_i in range(band_h):
        hr_i = y0 + local_i
        for hr_j in range(Ws):
            lr_x = f32((hr_j + 0.5) / cfg.scale)
            lr_y = f32((hr_i + 0.5) / cfg.scale)
            px = int(f32(lr_x / f32(cfg.tile_size)))
            py = int(f32(lr_y / f32(cfg.tile_size)))
            flowx = f32(flow[py, px, 0])
            flowy = f32(flow[py, px, 1])
            i_r = min(int(lr_y), lr_h - 1)
            j_r = min(int(lr_x), lr_w - 1)
            local_r = f32(rob[i_r, j_r])
            lr_mov_x = f32(lr_x + flowx)
            lr_mov_y = f32(lr_y + flowy)
            if not (
                lr_mov_x >= 0.0
                and lr_mov_x < f32(lr_w)
                and lr_mov_y >= 0.0
                and lr_mov_y < f32(lr_h)
            ):
                continue
            ixx = ixy = iyy = f32(0.0)
            if not cfg.iso:
                if cfg.bayer:
                    kmap_j = f32(lr_mov_x / 2.0 - 0.5)
                    kmap_i = f32(lr_mov_y / 2.0 - 0.5)
                else:
                    kmap_j = f32(lr_mov_x - 0.5)
                    kmap_i = f32(lr_mov_y - 0.5)
                ixx, ixy, iyy = interp_inv_cov(covs, kmap_i, kmap_j, True)
            center_j = int(lr_mov_x)
            center_i = int(lr_mov_y)
            lr_mov_j = f32(lr_mov_x - 0.5)
            lr_mov_i = f32(lr_mov_y - 0.5)
            val = [f32(0.0)] * 3
            acc = [f32(0.0)] * 3
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    j = center_j + dj
                    i = center_i + di
                    if not (0 <= j < lr_w and 0 <= i < lr_h):
                        continue
                    ch = cfa_ch(cfg, i, j)
                    c = f32(img[i, j])
                    dist_x = f32(f32(j) - lr_mov_j)
                    dist_y = f32(f32(i) - lr_mov_i)
                    if cfg.iso:
                        z = f32(2.0 * (dist_x * dist_x + dist_y * dist_y))
                    else:
                        z = f32(
                            ixx * dist_x * dist_x
                            + 2.0 * ixy * dist_x * dist_y
                            + iyy * dist_y * dist_y
                        )
                    z = f32(max(0.0, z))
                    w = exp_f(f32(-0.5 * z))
                    val[ch] = f32(val[ch] + w * local_r * c)
                    acc[ch] = f32(acc[ch] + w * local_r)
            for ch in range(cfg.nch):
                num[local_i, hr_j, ch] = f32(num[local_i, hr_j, ch] + val[ch])
                den[local_i, hr_j, ch] = f32(den[local_i, hr_j, ch] + acc[ch])


def accumulate_comp_metal(
    img, flow, covs, rob, num, den, y0: int, cfg: Cfg
) -> None:
    """Literal HHSRKernels.metal merge_accumulate_comp (post flow-OOB fix)."""
    band_h, Ws = num.shape[0], num.shape[1]
    lr_h, lr_w = img.shape
    for local_i in range(band_h):
        hr_i = y0 + local_i
        for hr_j in range(Ws):
            lr_x = f32((f32(hr_j) + 0.5) / cfg.scale)
            lr_y = f32((f32(hr_i) + 0.5) / cfg.scale)
            px = int(f32(lr_x / f32(cfg.tile_size)))
            py = int(f32(lr_y / f32(cfg.tile_size)))
            flowx = f32(flow[py, px, 0])
            flowy = f32(flow[py, px, 1])
            i_r = min(int(lr_y), lr_h - 1)
            j_r = min(int(lr_x), lr_w - 1)
            local_r = f32(rob[i_r, j_r])
            lr_mov_x = f32(lr_x + flowx)
            lr_mov_y = f32(lr_y + flowy)
            if not (
                lr_mov_x >= 0.0
                and lr_mov_x < f32(lr_w)
                and lr_mov_y >= 0.0
                and lr_mov_y < f32(lr_h)
            ):
                continue
            ixx = ixy = iyy = f32(0.0)
            if not cfg.iso:
                if cfg.bayer:
                    kmap_j = f32(lr_mov_x / 2.0 - 0.5)
                    kmap_i = f32(lr_mov_y / 2.0 - 0.5)
                else:
                    kmap_j = f32(lr_mov_x - 0.5)
                    kmap_i = f32(lr_mov_y - 0.5)
                ixx, ixy, iyy = interp_inv_cov(covs, kmap_i, kmap_j, True)
            center_j = int(lr_mov_x)
            center_i = int(lr_mov_y)
            lr_mov_j = f32(lr_mov_x - 0.5)
            lr_mov_i = f32(lr_mov_y - 0.5)
            val = [f32(0.0)] * 3
            acc = [f32(0.0)] * 3
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    j = center_j + dj
                    i = center_i + di
                    if not (0 <= j < lr_w and 0 <= i < lr_h):
                        continue
                    ch = cfa_ch(cfg, i, j)
                    c = f32(img[i, j])
                    dist_x = f32(f32(j) - lr_mov_j)
                    dist_y = f32(f32(i) - lr_mov_i)
                    if cfg.iso:
                        z = f32(2.0 * (dist_x * dist_x + dist_y * dist_y))
                    else:
                        z = f32(
                            ixx * dist_x * dist_x
                            + 2.0 * ixy * dist_x * dist_y
                            + iyy * dist_y * dist_y
                        )
                    z = f32(max(0.0, z))
                    w = exp_f(f32(-0.5 * z))
                    val[ch] = f32(val[ch] + w * local_r * c)
                    acc[ch] = f32(acc[ch] + w * local_r)
            for ch in range(cfg.nch):
                num[local_i, hr_j, ch] = f32(num[local_i, hr_j, ch] + val[ch])
                den[local_i, hr_j, ch] = f32(den[local_i, hr_j, ch] + acc[ch])


def accumulate_ref_cpu(
    img, covs, acc_rob: Optional[np.ndarray], num, den, y0: int, cfg: Cfg
) -> None:
    band_h, Ws = num.shape[0], num.shape[1]
    lr_h, lr_w = img.shape
    for local_i in range(band_h):
        hr_i = y0 + local_i
        for hr_j in range(Ws):
            coarse_x = f32(f32(hr_j) / cfg.scale)
            coarse_y = f32(f32(hr_i) / cfg.scale)
            local_acc_r = f32(0.0)
            additional_denoise_power = f32(1.0)
            rad = 1
            if cfg.robustness_denoise and acc_rob is not None:
                ay = min(round_half_away(coarse_y), acc_rob.shape[0] - 1)
                ax = min(round_half_away(coarse_x), acc_rob.shape[1] - 1)
                local_acc_r = f32(acc_rob[ay, ax])
                additional_denoise_power = (
                    cfg.max_multiplier
                    if local_acc_r <= cfg.max_frame_count
                    else f32(1.0)
                )
                rad = cfg.rad_max if local_acc_r <= cfg.max_frame_count else 1
            ixx = ixy = iyy = f32(0.0)
            if not cfg.iso:
                if cfg.bayer:
                    kmap_j = f32((coarse_x - 0.5) / 2.0)
                    kmap_i = f32((coarse_y - 0.5) / 2.0)
                else:
                    kmap_j, kmap_i = coarse_x, coarse_y
                ixx, ixy, iyy = interp_inv_cov(covs, kmap_i, kmap_j, False)
            center_j = round_half_away(coarse_x)
            center_i = round_half_away(coarse_y)
            val = [f32(0.0)] * 3
            acc = [f32(0.0)] * 3
            for di in range(-rad, rad + 1):
                for dj in range(-rad, rad + 1):
                    j = center_j + dj
                    i = center_i + di
                    if not (0 <= j < lr_w and 0 <= i < lr_h):
                        continue
                    ch = cfa_ch(cfg, i, j)
                    c = f32(img[i, j])
                    dist_x = f32(f32(j) - coarse_x)
                    dist_y = f32(f32(i) - coarse_y)
                    if cfg.iso:
                        y = f32(max(0.0, 2.0 * (dist_x * dist_x + dist_y * dist_y)))
                    else:
                        y = f32(
                            max(
                                0.0,
                                ixx * dist_x * dist_x
                                + 2.0 * ixy * dist_x * dist_y
                                + iyy * dist_y * dist_y,
                            )
                        )
                    y = f32(y / additional_denoise_power)
                    w = exp_f(f32(-0.5 * y))
                    val[ch] = f32(val[ch] + c * w)
                    acc[ch] = f32(acc[ch] + w)
            overwrite = (
                cfg.robustness_denoise
                and acc_rob is not None
                and local_acc_r < cfg.max_frame_count
            )
            for ch in range(cfg.nch):
                if overwrite:
                    num[local_i, hr_j, ch] = val[ch]
                    den[local_i, hr_j, ch] = acc[ch]
                else:
                    num[local_i, hr_j, ch] = f32(num[local_i, hr_j, ch] + val[ch])
                    den[local_i, hr_j, ch] = f32(den[local_i, hr_j, ch] + acc[ch])


def accumulate_ref_metal(
    img, covs, acc_rob: Optional[np.ndarray], num, den, y0: int, cfg: Cfg
) -> None:
    """Literal metal merge_accumulate_ref (robustness_denoise already gated like host)."""
    band_h, Ws = num.shape[0], num.shape[1]
    lr_h, lr_w = img.shape
    # Host sets robustness_denoise only when acc_rob present ΓÇö mirror that.
    denoise = cfg.robustness_denoise and acc_rob is not None
    for local_i in range(band_h):
        hr_i = y0 + local_i
        for hr_j in range(Ws):
            coarse_x = f32(f32(hr_j) / cfg.scale)
            coarse_y = f32(f32(hr_i) / cfg.scale)
            local_acc_r = f32(0.0)
            additional_denoise_power = f32(1.0)
            rad = 1
            if denoise:
                ay = min(round_half_away(coarse_y), acc_rob.shape[0] - 1)
                ax = min(round_half_away(coarse_x), acc_rob.shape[1] - 1)
                local_acc_r = f32(acc_rob[ay, ax])
                additional_denoise_power = (
                    cfg.max_multiplier
                    if local_acc_r <= cfg.max_frame_count
                    else f32(1.0)
                )
                rad = cfg.rad_max if local_acc_r <= cfg.max_frame_count else 1
            ixx = ixy = iyy = f32(0.0)
            if not cfg.iso:
                if cfg.bayer:
                    kmap_j = f32((coarse_x - 0.5) / 2.0)
                    kmap_i = f32((coarse_y - 0.5) / 2.0)
                else:
                    kmap_j, kmap_i = coarse_x, coarse_y
                ixx, ixy, iyy = interp_inv_cov(covs, kmap_i, kmap_j, False)
            center_j = round_half_away(coarse_x)
            center_i = round_half_away(coarse_y)
            val = [f32(0.0)] * 3
            acc = [f32(0.0)] * 3
            for di in range(-rad, rad + 1):
                for dj in range(-rad, rad + 1):
                    j = center_j + dj
                    i = center_i + di
                    if not (0 <= j < lr_w and 0 <= i < lr_h):
                        continue
                    ch = cfa_ch(cfg, i, j)
                    c = f32(img[i, j])
                    dist_x = f32(f32(j) - coarse_x)
                    dist_y = f32(f32(i) - coarse_y)
                    if cfg.iso:
                        y = f32(max(0.0, 2.0 * (dist_x * dist_x + dist_y * dist_y)))
                    else:
                        y = f32(
                            max(
                                0.0,
                                ixx * dist_x * dist_x
                                + 2.0 * ixy * dist_x * dist_y
                                + iyy * dist_y * dist_y,
                            )
                        )
                    y = f32(y / additional_denoise_power)
                    w = exp_f(f32(-0.5 * y))
                    val[ch] = f32(val[ch] + c * w)
                    acc[ch] = f32(acc[ch] + w)
            overwrite = denoise and (local_acc_r < cfg.max_frame_count)
            for ch in range(cfg.nch):
                if overwrite:
                    num[local_i, hr_j, ch] = val[ch]
                    den[local_i, hr_j, ch] = acc[ch]
                else:
                    num[local_i, hr_j, ch] = f32(num[local_i, hr_j, ch] + val[ch])
                    den[local_i, hr_j, ch] = f32(den[local_i, hr_j, ch] + acc[ch])


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def run_case(name: str, cfg: Cfg, scale: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cfg.scale = scale
    # Dimensions multiple of tile_size (pipeline pad).
    lr_h, lr_w = 64, 80
    tile = cfg.tile_size
    assert lr_h % tile == 0 and lr_w % tile == 0
    Hs = int(round(scale * lr_h))
    Ws = int(round(scale * lr_w))
    y0 = 7
    band_h = 11

    img = rng.random((lr_h, lr_w), dtype=np.float32)
    rob = rng.random((lr_h, lr_w), dtype=np.float32)
    flow = (rng.random((lr_h // tile, lr_w // tile, 2), dtype=np.float32) - 0.5) * 4.0
    cov_h, cov_w = lr_h // 2, lr_w // 2
    covs = rng.random((cov_h, cov_w, 4), dtype=np.float32)
    # Make SPD-ish diagonals
    covs[..., 0] = np.abs(covs[..., 0]) + 0.1
    covs[..., 3] = np.abs(covs[..., 3]) + 0.1
    covs[..., 1] *= 0.2
    covs[..., 2] = covs[..., 1]
    acc_rob = rng.random((lr_h, lr_w), dtype=np.float32) * 4.0

    num_c = np.zeros((band_h, Ws, cfg.nch), dtype=np.float32)
    den_c = np.zeros_like(num_c)
    num_m = np.zeros_like(num_c)
    den_m = np.zeros_like(num_c)

    accumulate_comp_cpu(img, flow, covs, rob, num_c, den_c, y0, cfg)
    accumulate_comp_metal(img, flow, covs, rob, num_m, den_m, y0, cfg)
    dnum = max_abs(num_c, num_m)
    dden = max_abs(den_c, den_m)
    if dnum != 0.0 or dden != 0.0:
        raise AssertionError(f"{name} comp mismatch num={dnum} den={dden}")

    num_c[:] = 0
    den_c[:] = 0
    num_m[:] = 0
    den_m[:] = 0
    accumulate_ref_cpu(img, covs, acc_rob if cfg.robustness_denoise else None, num_c, den_c, y0, cfg)
    accumulate_ref_metal(img, covs, acc_rob if cfg.robustness_denoise else None, num_m, den_m, y0, cfg)
    dnum = max_abs(num_c, num_m)
    dden = max_abs(den_c, den_m)
    if dnum != 0.0 or dden != 0.0:
        raise AssertionError(f"{name} ref mismatch num={dnum} den={dden}")

    print(f"OK  {name}  (scale={scale}, Hs={Hs}, band y0={y0})")


def main() -> int:
    cases = [
        ("bayer_steerable_1x", Cfg(bayer=True, iso=False, scale=1.0), 1.0, 1),
        ("bayer_steerable_2x", Cfg(bayer=True, iso=False, scale=2.0), 2.0, 2),
        ("bayer_iso_1x", Cfg(bayer=True, iso=True, scale=1.0), 1.0, 3),
        ("grey_steerable_2x", Cfg(bayer=False, iso=False, nch=1, scale=2.0), 2.0, 4),
        (
            "ref_acc_rob_denoise",
            Cfg(bayer=True, iso=False, robustness_denoise=True, rad_max=2),
            1.0,
            5,
        ),
        (
            "ref_acc_rob_denoise_2x",
            Cfg(bayer=True, iso=False, robustness_denoise=True, rad_max=2),
            2.0,
            6,
        ),
    ]
    for name, cfg, scale, seed in cases:
        run_case(name, cfg, scale, seed)
    print("All merge CPUΓåöMetal-semantics cases: bit-exact (float32).")
    print(
        "Note: on-device metal::precise::exp may differ by ulps from libm; "
        "algorithm/control-flow match is what this validates."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("FAIL:", e, file=sys.stderr)
        sys.exit(1)
