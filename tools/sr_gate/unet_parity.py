"""Check core/sr_gate_unet.cpp against the PyTorch model that was trained.

    python unet_parity.py --ckpt sr_gate_art.pt

The weights go to C++ (and later to Metal) as a flat blob indexed by hand. A
transposed filter, an off-by-one in the offsets, the skip concatenated in the
wrong order -- each of those produces a mask that looks entirely reasonable and
reports no error. So the forward pass is compared numerically against the
framework that fitted it.

Checked on REAL features from a synthesised burst as well as on random input:
random input exercises the arithmetic, real input exercises it in the value
range the network actually meets, where a wrong clamp or a saturating
intermediate would show and uniform noise would not.

Also sweeps the band boundary, since the C++ runs in bands with a halo and the
PyTorch version does not: if the halo is a row too small the error appears only
on the rows next to a band edge, which a single whole-image comparison with a
loose tolerance can hide.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate
import train_artifact as TA

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'unet_parity.exe')


def cpp_mask(feat, tau, beta):
    """feat (h, w, 20) -> mask (h, w) from the C++ implementation."""
    h, w, c = feat.shape
    tmp = tempfile.mkdtemp(prefix='srgu_')
    try:
        fi, fo = os.path.join(tmp, 'i.bin'), os.path.join(tmp, 'o.bin')
        with open(fi, 'wb') as f:
            f.write(struct.pack('<3i2f', h, w, c, tau, beta))
            f.write(np.ascontiguousarray(feat, dtype=np.float32).tobytes())
        r = subprocess.run([EXE, fi, fo], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit('C++ failed: %s%s' % (r.stdout, r.stderr))
        return np.fromfile(fo, dtype=np.float32).reshape(h, w)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def torch_mask(net, feat, tau, beta):
    x = torch.from_numpy(
        np.ascontiguousarray(np.transpose(feat, (2, 0, 1)),
                             dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        z = net(x)[:, 0]
        return TA.policy(torch.expm1(z.clamp(-8, 8)), tau, beta)[0].numpy()


def compare(name, a, b):
    d = np.abs(a - b)
    ok = d.max() < 2e-5
    print('  %-28s max %.3e  mean %.3e   %s'
          % (name, d.max(), d.mean(), 'OK' if ok else '*** MISMATCH ***'))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='sr_gate_art.pt')
    ap.add_argument('--data', default='data_ar')
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=0.5)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    net = gate.unet_from_checkpoint(torch.load(
        a.ckpt if os.path.isabs(a.ckpt) else os.path.join(here, a.ckpt),
        map_location='cpu', weights_only=False))
    print('C++ vs PyTorch, tau %.2f beta %.2f' % (a.tau, a.beta))

    ok = True
    rng = np.random.default_rng(0)
    for h, w in ((64, 64), (129, 97)):
        f = rng.standard_normal((h, w, 20)).astype(np.float32) * 0.3
        ok &= compare('random %dx%d' % (h, w),
                      cpp_mask(f, a.tau, a.beta), torch_mask(net, f, a.tau, a.beta))

    # Taller than one band (128) so the halo logic is exercised, and a size that
    # is not a multiple of it so the last band is short.
    for h, w in ((300, 64), (256, 64)):
        f = rng.standard_normal((h, w, 20)).astype(np.float32) * 0.3
        c, t = cpp_mask(f, a.tau, a.beta), torch_mask(net, f, a.tau, a.beta)
        ok &= compare('banded %dx%d' % (h, w), c, t)
        # Rows either side of the band seam, reported separately: a halo one row
        # short shows up here and nowhere else.
        for seam in (128, 256):
            if seam + 2 <= h:
                d = np.abs(c[seam - 2:seam + 2] - t[seam - 2:seam + 2]).max()
                print('      seam at row %-4d max %.3e' % (seam, d))

    # Real features, if a packed dataset is present.
    root = a.data if os.path.isabs(a.data) else os.path.join(here, a.data)
    fp = os.path.join(root, 'val_feat.npy')
    ip = os.path.join(root, 'val_img.npy')
    if os.path.exists(fp) and os.path.exists(ip):
        fe = np.load(fp, mmap_mode='r')
        im = np.load(ip, mmap_mode='r')
        for b in (0, 7):
            x = np.concatenate([np.asarray(fe[b, 0, :8], dtype=np.float32),
                                np.asarray(im[b, 0], dtype=np.float32)], axis=0)
            x = np.ascontiguousarray(np.transpose(x, (1, 2, 0)))
            ok &= compare('real burst %d' % b,
                          cpp_mask(x, a.tau, a.beta),
                          torch_mask(net, x, a.tau, a.beta))
    else:
        print('  (no packed dataset at %s -- real-feature check skipped)' % root)

    print('PARITY', 'OK' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
