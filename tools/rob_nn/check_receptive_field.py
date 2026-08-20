"""Measure the receptive field, and prove the strip halo covers it.

Three numbers get quoted about this network and they are easy to conflate:

  RADIUS      how far one output pixel reaches in each direction
  SIZE        the patch it sees, 2*radius + 1
  HALO        the extra rows a strip must carry to be exact, which must be >=
              RADIUS

An insufficient halo does not fail loudly. It produces horizontal seams every
kRobustnessNnStripRows rows, which are easy to miss by eye and look exactly
like model error. So this computes the radius from the layer parameters as
implemented, measures it empirically by gradient, and then bit-compares a
strip-processed plane against a whole-plane one.

Run it after any architecture change. If the bit-compare is not EXACT, the halo
in core/types.h is wrong.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rob import RobNet, IN_CH

HALO = 8       # kRobustnessNnHalo
STRIP = 192    # kRobustnessNnStripRows


def analytic_radius(model):
    """Sum of per-layer radii. All strides are 1, so radii simply add.

    A kxk convolution with dilation d has effective kernel 1 + d*(k-1) and
    therefore radius d*(k-1)/2. 1x1 convolutions contribute 0.
    """
    r = 0
    terms = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            assert m.stride == (1, 1), f"stride {m.stride} breaks the sum rule"
            k, d = m.kernel_size[0], m.dilation[0]
            ri = d * (k - 1) // 2
            r += ri
            terms.append(f"{k}x{k} d={d} -> {ri}")
    return r, terms


def empirical_radius(model, size=81):
    """Which inputs the centre output actually depends on, by gradient."""
    x = torch.zeros(1, IN_CH, size, size, requires_grad=True)
    y = model.logits(x)
    y[0, 0, size // 2, size // 2].backward()
    g = x.grad.abs().sum(1)[0].numpy()
    ys, xs = np.nonzero(g > 0)
    if len(ys) == 0:
        return 0, 0
    return int(max(abs(ys - size // 2).max(), abs(xs - size // 2).max())), int(len(ys))


def main():
    torch.manual_seed(0)
    model = RobNet()
    # A zero-initialised head would make every gradient zero and the empirical
    # measurement meaningless, so randomise it for this test only.
    torch.nn.init.normal_(model.net[-1].weight, std=0.5)
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.5)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.1)
    model.eval()

    r, terms = analytic_radius(model)
    print("layer radii: " + ", ".join(terms))
    print(f"analytic receptive-field RADIUS  {r} guide px  "
          f"({2 * r} raw px)")
    print(f"analytic receptive-field SIZE    {2 * r + 1} x {2 * r + 1} guide px  "
          f"({2 * (2 * r + 1)} raw px across)")
    er, n = empirical_radius(model)
    print(f"empirical radius by gradient     {er} guide px  "
          f"({n} of {(2*er+1)**2} pixels in the square have nonzero gradient)")
    ok_r = (er == r)
    print(f"  analytic == empirical: {'OK' if ok_r else 'MISMATCH'}")
    print(f"kRobustnessNnHalo = {HALO}, needs >= {r}: "
          f"{'OK' if HALO >= r else 'TOO SMALL'}")

    # The decisive test: does the strip loop reproduce whole-plane inference?
    # Windowing exactly as core/robustness.cpp does it -- every window the same
    # height and kept fully inside the image, so the convolutions' zero-padding
    # matches the whole-plane padding at the top and bottom edges.
    h, w = 3 * STRIP + 37, 64
    x = torch.randn(1, IN_CH, h, w)
    with torch.no_grad():
        whole = model(x)[0, 0].numpy()
    strips = np.zeros_like(whole)
    win = STRIP + 2 * HALO
    for y0 in range(0, h, STRIP):
        top = min(max(y0 - HALO, 0), h - win)
        with torch.no_grad():
            r_ = model(x[:, :, top:top + win])[0, 0].numpy()
        rows = min(STRIP, h - y0)
        strips[y0:y0 + rows] = r_[y0 - top:y0 - top + rows]
    diff = np.abs(whole - strips).max()
    exact = bool((whole == strips).all())
    print(f"\nstrip vs whole plane on {h}x{w}: max |diff| {diff:.3e}, "
          f"bit-identical {exact}")
    if not exact:
        rows = np.abs(whole - strips).max(axis=1)
        bad = np.nonzero(rows > 0)[0]
        print(f"  first differing rows: {bad[:12]}")
        print("  the halo does not cover the receptive field -- fix "
              "kRobustnessNnHalo in core/types.h")
    # And the counter-example, so the test is known to be able to fail: a halo
    # of 0 must produce seams. A test that passes for both is testing nothing.
    strips0 = np.zeros_like(whole)
    for y0 in range(0, h, STRIP):
        top = min(max(y0, 0), h - STRIP)
        with torch.no_grad():
            r_ = model(x[:, :, top:top + STRIP])[0, 0].numpy()
        rows = min(STRIP, h - y0)
        strips0[y0:y0 + rows] = r_[y0 - top:y0 - top + rows]
    print(f"control, halo 0: max |diff| {np.abs(whole - strips0).max():.3e} "
          "(must be large, or this test cannot detect a bad halo)")
    sys.exit(0 if (ok_r and exact and HALO >= r) else 1)


if __name__ == "__main__":
    main()
