"""Label correctness on freshly generated small samples (P = 64):
  - perfectly aligned bursts (exact tile vectors, pure translation) -> alignment error ~0, alignment target ~1, and
    the merge target keeps essentially every sample: real texture, flat areas AND aliased fine text (situations A, C)
  - deliberately corrupted frames -> samples displaced > 1 px on edges are mostly rejected, the same displacement in
    flat areas mostly kept (no universal displacement threshold)
  - include / exclude utility: a perfectly aligned frame helps (Delta > 0) almost everywhere
  - the label is the optimum of its own cost: R* costs no more than R = 1, R = 0 or the classical mask
"""
import json
import os
import tempfile
import unittest

import numpy as np
import torch

import common  # noqa: F401  (sys.path)
from mfsr import config as C, labels, merge as mg, sample


def _gen(category, idx, corrupt_prob=None, iso=None):
    over = {"patch": 64, "labels": {"iters": 120}}
    if corrupt_prob is not None:
        over["corrupt_prob"] = corrupt_prob             # every comparison frame gets at least one error kind
    if iso is not None:
        over["iso_mult"] = iso
    cfg = C._merge(C.DATA_DEFAULTS, over)
    d = tempfile.mkdtemp()
    fn = os.path.join(d, f"{category}.npz")
    sample.generate("val", 7000 + idx, cfg, fn, category=category, keep_utility=True)
    return np.load(fn), cfg


def _edge_flat(z):
    """Edges / flat areas RELATIVE TO THE NOISE (a structure below the noise cannot show a ghost): luma gradient of the
    HR target over the single-frame noise sigma, >= 2 = edge, <= 0.3 = flat."""
    meta = json.loads(str(z["meta"]))
    T = torch.from_numpy(z["T"].astype(np.float32))
    s = mg.single_frame_sig2(T, np.array(meta["prewhite"], np.float32), meta["noise_a"], meta["noise_b"]).sqrt()
    lum = (T * torch.tensor(mg.LUMA_W).view(3, 1, 1)).sum(0)
    g = (torch.sqrt(sum(d ** 2 for d in torch.gradient(lum))) / s).numpy()[::2, ::2]
    P1 = z["R0"].shape[-1]
    g = np.pad(g, ((0, P1 - g.shape[0]), (0, P1 - g.shape[1])), mode="edge")
    # flat = no structure anywhere a displaced sample can come from (9x9 raw neighbourhood): a reference pixel in a
    # flat spot whose displaced sample lands on a stroke DOES paint a ghost
    gmax = torch.nn.functional.max_pool2d(torch.from_numpy(g)[None, None], 9, 1, 4)[0, 0].numpy()
    return g >= 2.0, gmax <= 0.3


class Labels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.tex, cls.cfg = _gen("aligned_texture", 1)
        cls.txt, _ = _gen("aliasing", 2)
        cls.bad = None                                   # real scene, low ISO, every frame corrupted: first seed with
        for idx in (3, 4, 5):                            # enough displaced samples on visible edges AND in flat areas
            z, _ = _gen("aligned_texture", idx, corrupt_prob=1.0, iso=[1.0, 2.0])
            e = z["e1"].astype(np.float32)
            edge, flat = _edge_flat(z)
            big = (e > 0.75) & (e < 4.0)
            if (big & edge).sum() >= 20 and (big & flat).sum() >= 20:
                cls.bad = z
                break
        cls.bad = cls.bad if cls.bad is not None else z

    def test_aligned_samples_kept(self):
        for name, z in (("real texture", self.tex), ("aliased text", self.txt)):
            e, R, vis = z["e0"].astype(np.float32), z["R0"].astype(np.float32), z["vis"].astype(np.float32)
            with self.subTest(scene=name):
                self.assertLess(float(np.percentile(e, 99)), 0.05)
                ta = vis * np.exp(-e ** 2 / (2 * 0.25 ** 2))
                self.assertGreater(float(ta.mean()), 0.98)
                self.assertGreater(float((R >= 0.5).mean()), 0.99, "aligned samples labelled for rejection")

    def test_misaligned_edges_rejected_flat_kept(self):
        z = self.bad
        e, R = z["e1"].astype(np.float32), z["R1"].astype(np.float32)
        edge, flat = _edge_flat(z)
        big = (e > 0.75) & (e < 4.0)              # displaced, but not further than the flatness window
        me, mf = big & np.broadcast_to(edge, e.shape), big & np.broadcast_to(flat, e.shape)
        if me.sum() < 20 or mf.sum() < 20:
            self.skipTest("too few displaced samples drawn on edges / flat areas in this sample")
        self.assertGreater(float((R[me] < 0.5).mean()), 0.5)
        self.assertGreater(float((R[mf] >= 0.5).mean()), 0.8)

    def test_include_exclude_utility(self):
        D = self.tex["D0"].astype(np.float32)
        self.assertGreater(float((D > 0).mean()), 0.95)

    def test_label_is_optimum_of_its_cost(self):
        z = self.tex
        meta = json.loads(str(z["meta"]))
        P, m, scale, cfa = meta["P"], meta["m"], meta["scale"], tuple(tuple(r) for r in meta["cfa"])
        wb = np.array(meta["prewhite"], np.float32)
        raw = torch.from_numpy(z["raw"].astype(np.float32))
        clean = torch.from_numpy(z["clean"].astype(np.float32))
        var = sample.var_from_clean(clean, z["gain"], wb, cfa, meta["noise_a"], meta["noise_b"])
        T = torch.from_numpy(z["T"].astype(np.float32))
        sig2 = mg.single_frame_sig2(T, wb, meta["noise_a"], meta["noise_b"])
        hv = dict(flow=z["flow1"], H=z["H1"], has_h=z["hash1"])
        gtd = sample.dense_gt_xy(z["disp"].astype(np.float64), m, m, P, scale)
        parts = sample.merge_parts_xy(raw, clean, var, z["cov"].astype(np.float32), hv, scale, cfa, m, m, P,
                                      meta["tile"], gt_dense=gtd, vis=torch.from_numpy(z["vis"].astype(np.float32)))
        parts.g = mg.frame_gains(parts, T, sig2)
        cost = lambda R: mg.cost_map(parts, parts.fetch(R), T, sig2, 4.0, 0.02, "linear").mean().item()
        Rs = torch.from_numpy(z["R1"].astype(np.float32))
        r0 = torch.from_numpy(z["r01"].astype(np.float32)).repeat_interleave(2, -2).repeat_interleave(2, -1)
        r0 = r0[:, m:m + P + 1, m:m + P + 1]
        cs = cost(Rs)
        for name, R in (("ones", torch.ones_like(Rs)), ("zeros", torch.zeros_like(Rs)), ("classical", r0)):
            with self.subTest(other=name):
                self.assertLessEqual(cs, cost(R) * 1.001 + 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
