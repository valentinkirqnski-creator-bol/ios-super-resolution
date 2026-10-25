"""The PyTorch merge replica (mfsr/merge.py) against the app's own C++ merge (merge.cpp via conf_harness, R = 1):
reference and comparison contributions, tile flow, with and without the global homography, with a deliberate
homography error (rotation inside tiles) and with large tile errors that push samples out of the frame."""
import unittest

import numpy as np
import torch

from common import burst
from mfsr import corrupt, harness, merge as mg


def _run(seed, flags, rot=0.0, pert_kind=None):
    f, cfa, wb, sc = burst(seed=seed, S=128, n=4, rot=rot, tremor=2.0)
    fr = f.render(seed)
    d, _ = f.correspondence_grid(0, 0, 128, 128)
    gt = d.permute(0, 2, 3, 1).float().numpy()
    N = f.N
    pert = np.zeros((N, 128, 128, 2), np.float32)
    hp = np.tile(np.eye(3), (N, 1, 1))
    if pert_kind:
        rng = np.random.default_rng(seed + 99)
        for k in range(1, N):
            p, h = corrupt.draw(pert_kind, rng, 128)
            pert[k], hp[k] = p, h
    h = harness.run(fr["raw"].numpy(), gt, pert, hp, cfa, 2.0, sc["noise_a"], sc["noise_b"], wb,
                    flags=flags | harness.FLAG_PARITY, seed=seed)
    raw = fr["raw"]
    zeros = torch.zeros_like(raw[0])
    Ho = Wo = 256
    out = []
    Aref, _, Bref, _ = mg.ref_contrib(raw[0], raw[0], zeros, torch.as_tensor(h["cov_ref"]), 2.0, cfa, 0, 0, Ho, Wo)
    out.append(("ref A", Aref, h["Aref"]))
    out.append(("ref B", Bref, h["Bref"]))
    for k in range(1, N):
        A, _, B, _ = mg.comp_contrib(raw[k], raw[k], zeros, torch.as_tensor(h["flow"][k - 1]),
                                     torch.as_tensor(h["H"][k - 1]), bool(h["has_h"][k - 1] > 0.5),
                                     torch.as_tensor(h["cov"][k - 1]), h["tile"], 2.0, cfa, 0, 0, Ho, Wo)
        out.append((f"comp{k} A", A, h["A"][k - 1]))
        out.append((f"comp{k} B", B, h["B"][k - 1]))
    return out, h


class MergeParity(unittest.TestCase):
    def check(self, out, tol=2e-5):
        for name, py, cpp in out:
            c = torch.from_numpy(cpp).permute(2, 0, 1)
            err = (py - c).abs().max().item()
            scale = max(c.abs().max().item(), 1e-6)
            with self.subTest(part=name):
                self.assertLess(err / scale, tol, f"{name}: max abs err {err:.3g} (scale {scale:.3g})")

    def test_tile_flow(self):
        out, h = _run(1, 0)
        self.check(out)

    def test_homography(self):
        out, h = _run(2, harness.FLAG_HOMOG, rot=1.2)
        self.assertTrue((h["has_h"] > 0.5).all())
        self.check(out)

    def test_rotation_error_and_large_offsets(self):
        out, _ = _run(3, 0, pert_kind="rotation")
        self.check(out)
        out, _ = _run(4, 0, pert_kind="tile_large")
        self.check(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
