"""One synthetic burst end to end: scene -> RAW frames -> real (imperfect)
alignment -> optional deliberate corruption -> geometric + utility labels."""
import numpy as np
import torch

from . import scenarios
from .align import TileAligner, densify
from .burst import Burst
from .corrupt import corrupt
from .labels import geometric_label, utility_labels


def make_example(seed, scenario=None, size=256, aligner=None, scene_dir=None, out_scale=None,
                 corrupt_override=None):
    rng = np.random.default_rng(seed)
    sc = scenarios.sample(rng, scenario)
    tile = int(rng.choice([8, 16]))
    out_scale = out_scale or int(rng.choice([1, 2], p=[0.6, 0.4]))
    b = Burst(sc, rng, Hr=size, Wr=size, scene_dir=scene_dir)
    d = b.render(out_scale=out_scale, seed=rng.integers(1 << 31))
    N = sc["n_frames"]
    al = aligner or TileAligner(tile=tile)
    al.tile = tile
    grey = d["grey"]
    Hg, Wg = grey.shape[-2:]
    flows = torch.zeros(N, 2, Hg, Wg)
    corrupted = [False] * N
    ops = [[] for _ in range(N)]
    dmode = "nearest" if rng.random() < 0.75 else "bilinear"
    cp = sc["corrupt_prob"] if corrupt_override is None else corrupt_override
    for k in range(1, N):
        tf, _ = al(grey[0], grey[k])
        if rng.random() < cp:
            tf, ops[k] = corrupt(tf, grey[0], tile, rng)
            corrupted[k] = True
        flows[k] = densify(tf, Hg, Wg, tile, dmode)
    geo = torch.ones(N, Hg, Wg)
    err = torch.zeros(N, Hg, Wg)
    for k in range(1, N):
        geo[k], err[k] = geometric_label(d["grey_clean"][k], d["grey_clean"][0], flows[k], d["gt_flow"][k],
                                         d["vis"][k], d["noise_a"], d["noise_b"])
    sigma_k = float(rng.uniform(0.5, 1.0))
    util = utility_labels(d["raw"], flows, geo, d["gt"], d["noise_a"], d["noise_b"], out_scale, sigma_k)
    return dict(sc=sc, tile=tile, out_scale=out_scale, sigma_k=sigma_k, flows=flows, geo=geo, util=util,
                err=err, corrupted=corrupted, ops=ops, **d)


def to_record(ex, full=False):
    """Compact training record (grey-res), plus RAW frames + GT for the merge loss if full."""
    extra = {}
    if full:
        extra = dict(raw=ex["raw"].numpy().astype(np.float16), gt=ex["gt"].numpy().astype(np.float16),
                     out_scale=np.int32(ex["out_scale"]), sigma_k=np.float32(ex["sigma_k"]),
                     noise_a_raw=np.float32(ex["noise_a"]), noise_b_raw=np.float32(ex["noise_b"]))
    return dict(**extra,
        grey=ex["grey"].numpy().astype(np.float16), flows=ex["flows"].numpy().astype(np.float32),
        geo=ex["geo"].numpy().astype(np.float16), util=ex["util"].numpy().astype(np.float16),
        vis=ex["vis"].numpy().astype(np.float16), err=ex["err"].numpy().astype(np.float16),
        noise_a=np.float32(ex["noise_a"] / 4.0), noise_b=np.float32(ex["noise_b"] / 4.0),  # grey domain
        tile=np.int32(ex["tile"]), scenario=ex["sc"]["name"], corrupted=np.array(ex["corrupted"]),
    )
