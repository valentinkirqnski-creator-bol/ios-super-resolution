#!/usr/bin/env python3
"""Convert a Handheld-MFSR-1.4 noise LUT (.npz from monte_carlo.py) into the
raw .bin the C++ robustness path loads (NoiseLut14 / active_noise_lut14).

The .npz is deflate-compressed, so the app avoids parsing it directly (that
would need zlib in the noise path). This one-time conversion produces a tiny
flat binary the port reads with plain fread.

Binary layout (little-endian):
    char   magic[4]      = "N14L"
    int32  bins                         (must be 1001 to match the guide bins)
    float32 alpha_rgbg[4]
    float32 beta_rgbg[4]
    float32 sigma_noise_sq[bins]        E[Σ_c var | mean sqrt-brightness]
    float32 d_noise_sq[bins]            E[Σ_c Δμ² | mean sqrt-brightness]

Usage:
    python tools/npz_to_noise_lut_bin.py in.npz out/noise_lut.bin
Then place noise_lut.bin where the app looks (iOS:
    ~/Documents/noise_curves/noise_lut.bin), or set HHSR_NOISE_LUT14=<path>.
"""
import struct
import sys

import numpy as np


def convert(npz_path: str, bin_path: str) -> None:
    z = np.load(npz_path)
    if str(z.get("transform", "")) and "sqrt" not in str(z["transform"]):
        print(f"warning: transform={z['transform']!r} is not a sqrt-domain LUT; "
              "the port indexes by the sqrt-domain guide mean.", file=sys.stderr)
    bins = int(z["bins"]) if "bins" in z else int(np.asarray(z["sigma_noise_sq"]).size)
    sigma = np.asarray(z["sigma_noise_sq"], dtype="<f4")
    d = np.asarray(z["d_noise_sq"], dtype="<f4")
    if sigma.size != bins or d.size != bins:
        raise SystemExit(f"curve length {sigma.size}/{d.size} != bins {bins}")
    alpha = np.asarray(z["alpha_rgbg"], dtype="<f4").ravel()
    beta = np.asarray(z["beta_rgbg"], dtype="<f4").ravel()
    if alpha.size != 4 or beta.size != 4:
        raise SystemExit("alpha_rgbg/beta_rgbg must have 4 entries (R,G1,B,G2)")
    with open(bin_path, "wb") as f:
        f.write(b"N14L")
        f.write(struct.pack("<i", bins))
        f.write(alpha.tobytes())
        f.write(beta.tobytes())
        f.write(sigma.tobytes())
        f.write(d.tobytes())
    print(f"wrote {bin_path}: bins={bins} alpha={alpha} beta={beta}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: npz_to_noise_lut_bin.py <in.npz> <out.bin>")
    convert(sys.argv[1], sys.argv[2])
