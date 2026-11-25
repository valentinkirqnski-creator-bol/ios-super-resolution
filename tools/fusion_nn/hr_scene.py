"""12 MP HR RGB target from one DNG, in the app's raw units.

    python hr_scene.py <dng> <app_raw.f32> <app_raw.meta> <out.npy>

LibRaw (rawpy) DCB demosaic, linear, camera white balance, camera RGB (no colour matrix),
no flip. Then a per-channel gain so the HR image is in exactly the units of the app's
loader (black-subtracted, prewhitened, normalised): fitted on 2x2 means against the app's
own raw (app_raw.f32 from dng_dump). Saved as float16 [3, H, W].
"""
import sys

import numpy as np
import rawpy


def main():
    dng, f32, meta, out = sys.argv[1:5]
    m = open(meta).read().split()
    H, W = int(m[0]), int(m[1])
    cfa = [[int(m[2]), int(m[3])], [int(m[4]), int(m[5])]]
    with rawpy.imread(dng) as r:
        rgb = r.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.DCB, output_color=rawpy.ColorSpace.raw,
                            gamma=(1, 1), no_auto_bright=True, use_camera_wb=True, output_bps=16, user_flip=0,
                            half_size=False, highlight_mode=rawpy.HighlightMode.Clip).astype(np.float32)
    print("rawpy", rgb.shape, "app raw", (H, W))
    assert rgb.shape[:2] == (H, W), "size mismatch"
    raw = np.fromfile(f32, np.float32).reshape(H, W)
    # app raw 2x2 quad -> per channel mean; HR -> the same quad means of its own channel
    hr = np.zeros((3, H, W), np.float32)
    for c in range(3):
        a = []
        b = []
        for dy in (0, 1):
            for dx in (0, 1):
                if cfa[dy][dx] == c:
                    a.append(raw[dy::2, dx::2])
                    b.append(rgb[dy::2, dx::2, c])
        a = np.mean(a, 0).ravel()
        b = np.mean(b, 0).ravel()
        sel = (a > 0.01) & (a < 0.8) & (b > 0) & (b < 60000)
        g = float(np.sum(a[sel] * b[sel]) / np.sum(b[sel] * b[sel]))
        hr[c] = rgb[..., c] * g
        res = np.abs(a[sel] - b[sel] * g) / a[sel]
        print("channel %d gain %.4e  median rel. diff on raw sites %.4f" % (c, g, np.median(res)))
    np.save(out, hr.astype(np.float16))
    print("saved", out, hr.shape, "mean", hr.mean((1, 2)))


if __name__ == "__main__":
    main()
