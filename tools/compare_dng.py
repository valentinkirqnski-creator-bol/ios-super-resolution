#!/usr/bin/env python3
"""
Perceptual-equivalence gate for HandheldSR DNG output.

Optimization work (threadgroup reductions, fast::exp, reordered float sums)
changes results in the last few ULPs. That is fine as long as the image is
indistinguishable to the eye at maximum magnification -- but "indistinguishable"
has to be measured, not eyeballed. This compares two LinearRaw DNGs and reports
whether the difference is below the visibility floor.

Default gate: max per-channel difference <= 2 LSB (of 65535) and PSNR >= 80 dB.
For scale, one step of 8-bit display quantization is 257 LSB at 16-bit scale,
so a 2 LSB bound is ~128x below a single 8-bit increment.

Run: python tools/compare_dng.py reference.dng candidate.dng
Exit 0 iff the candidate is within tolerance.
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib

import numpy as np

T_SHORT, T_LONG = 3, 4


def _read_ifd_values(buf: bytes, type_: int, count: int, value_off: int):
    """Return the tag's values, following the offset when they don't fit inline."""
    size = {T_SHORT: 2, T_LONG: 4}.get(type_)
    if size is None:
        return []
    total = size * count
    if total <= 4:
        raw = struct.pack("<I", value_off)[:total]
    else:
        if value_off + total > len(buf):
            raise ValueError("tag payload runs past end of file")
        raw = buf[value_off:value_off + total]
    fmt = "<%d%s" % (count, "H" if size == 2 else "I")
    return list(struct.unpack(fmt, raw))


def load_linear_dng(path: str) -> np.ndarray:
    """Decode a HandheldSR LinearRaw DNG to a (H, W, 3) uint16 array."""
    with open(path, "rb") as f:
        buf = f.read()

    if len(buf) < 16 or buf[0:2] != b"II" or struct.unpack_from("<H", buf, 2)[0] != 42:
        raise ValueError("%s: not a little-endian TIFF/DNG" % path)

    ifd_off = struct.unpack_from("<I", buf, 4)[0]
    nent = struct.unpack_from("<H", buf, ifd_off)[0]

    tags = {}
    for i in range(nent):
        e = ifd_off + 2 + i * 12
        tag, type_, count = struct.unpack_from("<HHI", buf, e)
        value_off = struct.unpack_from("<I", buf, e + 8)[0]
        tags[tag] = (type_, count, value_off)

    def scalar(tag, default=None):
        if tag not in tags:
            return default
        type_, count, value_off = tags[tag]
        vals = _read_ifd_values(buf, type_, count, value_off)
        return vals[0] if vals else default

    def array(tag):
        if tag not in tags:
            return []
        return _read_ifd_values(buf, *tags[tag])

    width = scalar(256)
    height = scalar(257)
    compression = scalar(259, 1)
    spp = scalar(277, 3)
    predictor = scalar(317, 1)
    rows_per_strip = scalar(278, height)

    if not width or not height:
        raise ValueError("%s: missing ImageWidth/ImageLength" % path)
    if spp != 3:
        raise ValueError("%s: expected SamplesPerPixel=3, got %s" % (path, spp))
    if compression not in (1, 8):
        raise ValueError("%s: unsupported Compression=%s" % (path, compression))

    offsets = array(273)
    counts = array(279)
    if not offsets:
        raise ValueError("%s: missing StripOffsets" % path)
    if not counts:
        counts = [width * height * 3 * 2]

    row_bytes = width * 3 * 2
    out = bytearray()
    for idx, off in enumerate(offsets):
        nbytes = counts[idx] if idx < len(counts) else 0
        chunk = buf[off:off + nbytes]
        if compression == 8:
            chunk = zlib.decompress(chunk)
        else:
            # Uncompressed: the final strip may be short if it is partial.
            rows_here = min(rows_per_strip, height - idx * rows_per_strip)
            chunk = chunk[:rows_here * row_bytes]
        out += chunk

    expected = height * row_bytes
    if len(out) != expected:
        raise ValueError("%s: decoded %d bytes, expected %d" % (path, len(out), expected))

    img = np.frombuffer(bytes(out), dtype="<u2").reshape(height, width, 3)
    if predictor == 2:
        img = np.cumsum(img.astype(np.uint32), axis=1).astype(np.uint16)
    return img


def compare(ref: np.ndarray, cand: np.ndarray, max_lsb: int, min_psnr: float) -> bool:
    if ref.shape != cand.shape:
        print("FAIL: shape %s vs %s" % (ref.shape, cand.shape))
        return False

    a = ref.astype(np.int32)
    b = cand.astype(np.int32)
    diff = np.abs(a - b)

    total = diff.size
    ndiff = int(np.count_nonzero(diff))
    max_d = int(diff.max())
    mean_d = float(diff.mean())
    mse = float((diff.astype(np.float64) ** 2).mean())
    psnr = float("inf") if mse == 0.0 else 10.0 * np.log10((65535.0 ** 2) / mse)

    print("resolution        %d x %d" % (ref.shape[1], ref.shape[0]))
    print("samples           %d" % total)
    print("differing         %d (%.4f%%)" % (ndiff, 100.0 * ndiff / total))
    print("max |diff|        %d LSB  (of 65535)" % max_d)
    print("mean |diff|       %.4f LSB" % mean_d)
    print("PSNR              %s dB" % ("inf" if psnr == float("inf") else "%.2f" % psnr))
    # One 8-bit display step is 257 LSB at 16-bit scale.
    print("vs 8-bit step     max diff is %.1fx below one 8-bit increment"
          % (257.0 / max_d) if max_d else "vs 8-bit step     exact match")

    ok = (max_d <= max_lsb) and (psnr >= min_psnr)
    if ok:
        if max_d == 0:
            print("\nPASS: bit-identical")
        else:
            print("\nPASS: within tolerance (<= %d LSB, >= %.0f dB) - not visible"
                  % (max_lsb, min_psnr))
    else:
        print("\nFAIL: exceeds tolerance (max %d LSB > %d, or PSNR < %.0f dB)"
              % (max_d, max_lsb, min_psnr))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--max-lsb", type=int, default=2,
                    help="max allowed per-channel difference in 16-bit LSB (default 2)")
    ap.add_argument("--min-psnr", type=float, default=80.0,
                    help="minimum allowed PSNR in dB (default 80)")
    args = ap.parse_args()

    try:
        ref = load_linear_dng(args.reference)
        cand = load_linear_dng(args.candidate)
    except (OSError, ValueError, zlib.error) as exc:
        print("error: %s" % exc)
        return 2

    return 0 if compare(ref, cand, args.max_lsb, args.min_psnr) else 1


if __name__ == "__main__":
    sys.exit(main())
