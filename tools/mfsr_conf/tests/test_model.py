"""Confidence-map dimensions and the cross-frame fusion's contract."""
import glob
import os
import unittest

import torch

import common  # noqa: F401
from mfsr import model as M
from mfsr.data import Bursts, collate
from mfsr.inputs import NG, NR


class Model(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = M.ConfNet(ch=8, fusion=True).eval()
        with torch.no_grad():                         # non-trivial refinement head so the raw-res path is exercised
            self.net.u2.weight.normal_(0, 0.1)

    def inputs(self, B=2, K=5, h=24):
        xg = torch.randn(B, K, NG, h, h)
        xr = torch.randn(B, K, NR, 2 * h, 2 * h)
        mask = torch.ones(B, K)
        mask[1, 3:] = 0.0
        return xg, xr, mask

    def test_shapes_and_range(self):
        xg, xr, mask = self.inputs()
        out = self.net(xg, xr, mask)
        self.assertEqual(tuple(out.shape), (2, 5, 2, 48, 48))      # per frame, per RAW pixel: (alignment, merge)
        R = torch.sigmoid(out[:, :, 1])
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(((R >= 0) & (R <= 1)).all())

    def test_frame_permutation_equivariance(self):
        xg, xr, mask = self.inputs()
        mask[:] = 1.0
        out = self.net(xg, xr, mask)
        perm = torch.tensor([3, 0, 4, 1, 2])
        out_p = self.net(xg[:, perm], xr[:, perm], mask[:, perm])
        self.assertTrue(torch.allclose(out[:, perm], out_p, atol=1e-5))

    def test_padded_frames_do_not_leak(self):
        xg, xr, mask = self.inputs()
        out = self.net(xg, xr, mask)
        xg2, xr2 = xg.clone(), xr.clone()
        xg2[1, 3:] += 100.0
        xr2[1, 3:] -= 50.0
        out2 = self.net(xg2, xr2, mask)
        self.assertTrue(torch.allclose(out[1, :3], out2[1, :3], atol=1e-5))

    def test_maps_match_labels_on_real_data(self):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "d1")
        if not glob.glob(os.path.join(root, "val", "*.npz")):
            self.skipTest("no generated data")
        ds = Bursts(root, "val", crop=64, halo=32, n_crops=1, random_crop=True, limit=1)
        b = collate([ds[0]])
        out = self.net(b["xg"], b["xr"], b["mask"])
        Pc1 = b["R"].shape[-1]
        self.assertEqual(out.shape[-1], 64 + 2 * 32)
        self.assertEqual(out[..., 32:32 + Pc1, 32:32 + Pc1].shape[-2:], b["R"].shape[-2:])
        self.assertEqual(b["R"].shape[-1], 64 + 1)                 # the raw pixels the merge reads for the crop


if __name__ == "__main__":
    unittest.main(verbosity=2)
