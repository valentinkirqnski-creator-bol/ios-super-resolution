"""Transformation geometry, flow direction / units, the HR-target pixel convention and the mask fetch rule."""
import unittest

import numpy as np
import torch

from common import burst
from mfsr import harness, merge as mg


def _pts(S, m):
    yy, xx = torch.meshgrid(torch.arange(m, S - m, dtype=torch.float64), torch.arange(m, S - m, dtype=torch.float64),
                            indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)


class Geometry(unittest.TestCase):
    def test_translation_correspondence_is_exact(self):
        """Pure camera translation: the scene point the reference sees at p is what frame k sees at p + d_k(p), d is
        one constant vector, and the opposite sign is clearly wrong (direction convention)."""
        f, *_ = burst(seed=11, S=128, n=4, tremor=2.5)
        p = _pts(128, 24)
        d, vis = f.correspondence(p)
        r0 = f.radiance(0, p)
        for k in range(1, f.N):
            self.assertLess((d[k] - d[k].mean(1, keepdim=True)).abs().max().item(), 1e-9)
            good = (f.radiance(k, p + d[k].T) - r0).abs().max().item()
            bad = (f.radiance(k, p - d[k].T) - r0).abs().max().item()
            self.assertLess(good, 1e-5)
            if d[k].abs().max() > 0.3:
                self.assertGreater(bad, 50 * good + 1e-4)
            self.assertTrue((vis[k] > 0.99).all())

    def test_rotation_correspondence_point_sampling(self):
        """Camera roll + zoom: with point sampling (aperture n = 1) radiance(k, p + d_k(p)) == radiance(0, p)."""
        f, *_ = burst(seed=12, S=128, n=3, rot=1.5, zoom=0.004, aperture=1)
        p = _pts(128, 24)
        d, _ = f.correspondence(p)
        r0 = f.radiance(0, p)
        for k in range(1, f.N):
            self.assertLess((f.radiance(k, p + d[k].T) - r0).abs().max().item(), 1e-5)
            self.assertGreater((d[k] - d[k].mean(1, keepdim=True)).abs().max().item(), 0.05)   # really varies

    def test_app_flow_direction_and_units(self):
        """The app's aligner, run on a low-noise translated burst, returns tile vectors equal to our exact
        displacement (same sign, raw px); oracle mode returns it exactly."""
        f, cfa, wb, sc = burst(seed=13, S=256, n=4, tremor=3.0, iso=1.0)
        fr = f.render(13)
        d, _ = f.correspondence_grid(0, 0, 256, 256)
        gt = d.permute(0, 2, 3, 1).float().numpy()
        N = f.N
        z, hp = np.zeros((N, 256, 256, 2), np.float32), np.tile(np.eye(3), (N, 1, 1))
        # tolerance 0.5 px: this checks direction and units (a sign error shows up as ~2|d| = 7 px, guide-vs-raw px
        # as ~|d| = 3.7 px). Accuracy is NOT asserted: the app's aligner sometimes leaves a whole frame at the
        # integer position (measured: 0.26 px uniform error on one frame of this burst), which is real behaviour.
        for flags, tol in ((0, 0.5), (harness.FLAG_ORACLE, 1e-6)):
            h = harness.run(fr["raw"].numpy(), gt, z, hp, cfa, 2.0, sc["noise_a"], sc["noise_b"], wb, flags=flags)
            t = h["tile"]
            for k in range(1, N):
                ny, nx = h["flow"].shape[1:3]
                errs = []
                # interior tiles only: the app's aligner returns wrong vectors in a fixed band of ~1 tile at the
                # left and ~3 tiles at the bottom / right frame border (measured, see README); mfsr keeps its labelled
                # patch out of that band with asymmetric margins
                for ty in range(1, ny - 3):
                    for tx in range(1, nx - 3):
                        cy, cx = min(255, int((ty + 0.5) * t)), min(255, int((tx + 0.5) * t))
                        errs.append(np.abs(h["flow"][k - 1, ty, tx] - gt[k, cy, cx]).max())
                with self.subTest(flags=flags, frame=k):
                    self.assertLess(float(np.median(errs)), tol)

    def test_hr_target_matches_merge_output_grid(self):
        """With exact correspondences the merged noise-free burst agrees with T on the merge's own output grid
        (o / s) clearly better than with a target a quarter raw pixel off (the (o + 0.5) / s - 0.5 convention)."""
        f, cfa, wb, sc = burst(seed=14, S=128, n=8, tremor=2.0, iso=1.0, optics=0.5)
        fr = f.render(14)
        m, P = 24, 80
        d, _ = f.correspondence_grid(0, 0, 128, 128)
        gt = d.permute(0, 2, 3, 1).float().numpy()
        N = f.N
        z, hp = np.zeros((N, 128, 128, 2), np.float32), np.tile(np.eye(3), (N, 1, 1))
        h = harness.run(fr["raw"].numpy(), gt, z, hp, cfa, 2.0, sc["noise_a"], sc["noise_b"], wb, flags=harness.FLAG_ORACLE)
        from mfsr.sample import dense_gt, merge_parts
        cov = np.concatenate([h["cov_ref"][None], h["cov"]], 0)
        dense = dense_gt(d[1:, :, m:m + P + 1, m:m + P + 1].numpy(), m, P, 2.0)
        hv = dict(flow=h["flow"], H=h["H"], has_h=h["has_h"])
        clean = fr["clean"]
        parts = merge_parts(clean, clean, torch.zeros_like(clean), cov, hv, 2.0, cfa, m, P, h["tile"], dense=dense)
        M = parts.merged(parts.fetch(torch.ones(N - 1, P + 1, P + 1)), clean=True)
        T = f.hr_target(2 * m, 2 * m, 2 * P, 2 * P, 2.0)
        T_off = f.hr_target(2 * m - 0.5, 2 * m - 0.5, 2 * P, 2 * P, 2.0)
        e_ok = ((M - T)[:, 4:-4, 4:-4] ** 2).mean().item()
        e_off = ((M - T_off)[:, 4:-4, 4:-4] ** 2).mean().item()
        self.assertLess(e_ok, 0.7 * e_off, f"mse vs T {e_ok:.3g}, vs quarter-pixel-shifted T {e_off:.3g}")

    def test_mask_fetch_index(self):
        """merge.cpp reads a raw-resolution mask at floor(o / s + 0.5): output pixels 2p-1 and 2p read raw p."""
        fy, fx = mg.fetch_index(0, 0, 9, 9, 2.0, 0, 0, 6, 6)
        self.assertEqual(fx[0].tolist(), [0, 1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual(fy[:, 0].tolist(), [0, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_recombination_is_the_merge(self):
        """The app's merge run WITH a random raw-resolution R map (C++) equals the recombination of its R = 1
        contributions with that map fetched by the nearest rule (Python): the identity labels and the loss rest on."""
        f, cfa, wb, sc = burst(seed=15, S=128, n=4, tremor=2.0, rot=0.8)
        fr = f.render(15)
        d, _ = f.correspondence_grid(0, 0, 128, 128)
        gt = d.permute(0, 2, 3, 1).float().numpy()
        N = f.N
        z, hp = np.zeros((N, 128, 128, 2), np.float32), np.tile(np.eye(3), (N, 1, 1))
        rng = np.random.default_rng(5)
        rmap = rng.random((N - 1, 128, 128)).astype(np.float32)
        rmap[:, 40:70, 30:90] = 0.0                                   # a hard rejection block too
        one = harness.run(fr["raw"].numpy(), gt, z, hp, cfa, 2.0, sc["noise_a"], sc["noise_b"], wb,
                          flags=harness.FLAG_PARITY)
        withr = harness.run(fr["raw"].numpy(), gt, z, hp, cfa, 2.0, sc["noise_a"], sc["noise_b"], wb,
                            flags=harness.FLAG_PARITY | harness.FLAG_RMAP, rmap=rmap)
        fy, fx = mg.fetch_index(0, 0, 256, 256, 2.0, 0, 0, 128, 128, strict=False)
        for k in range(N - 1):
            r = torch.from_numpy(rmap[k])[fy, fx][..., None].numpy()
            for X in ("A", "B"):
                rec = r * one[X][k]
                err = np.abs(rec - withr[X][k]).max() / max(np.abs(withr[X][k]).max(), 1e-6)
                with self.subTest(frame=k, part=X):
                    self.assertLess(err, 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
