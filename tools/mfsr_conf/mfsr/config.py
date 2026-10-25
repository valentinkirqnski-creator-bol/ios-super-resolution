"""YAML configuration with defaults. A run's effective config is saved next to its outputs, so every result can be
reproduced from it (plus the code at the recorded git revision)."""
import copy
import hashlib
import json
import os
import subprocess

import yaml

from . import TOOL, ROOT

DATA_DEFAULTS = {
    "name": "d1",
    "seed": 1,
    "patch": 128,            # raw px of the labelled central patch (64 / 128 / 256)
    "margin": 48,            # raw px above / left of it: warp room, network halo (>= halo + 16)
    "margin_hi": 112,        # raw px below / right of it: the app's aligner is wrong in a ~3-tile band at the bottom /
                             # right frame border (and ~1 tile at the left), so the patch must stay clear of it
    "scale": 2.0,            # output / sensor (12 MP -> 48 MP)
    "tex_per_raw": 2.0,      # real-scene texels (one RGGB quad of the source DNG) per simulated sensor pixel
    "tex_per_raw_procedural": 4.0,
    "aperture": {"n": 3, "fill": 1.0},   # n x n sub-samples over the pixel footprint (box integration)
    "frames": [4, 8],
    "iso_mult": [1.0, 30.0],             # x the source DNG's own noise profile (log-uniform)
    "exposure_match_residual": 0.01,     # app's metadata gain matching: residual relative error (sigma)
    "scenes": {
        "train": ["APC_1340", "APC_1342", "HDRP_N000", "APC_1258"],
        "val": ["APC_1211"],
        "test": ["APC_1278"],
        "excluded": ["APC_1186", "APC_1294"],
    },
    "counts": {"train": 400, "val": 40, "test": 108},
    "categories": {
        "aligned_texture": 1.0, "aliasing": 1.0, "edge_slip": 1.5, "repetitive": 0.8, "flat_noisy": 1.0,
        "rotation": 1.5, "moving": 1.0, "exposure": 0.7, "saturation": 0.7,
    },
    "corrupt_prob": 0.7,     # per comparison frame, in the corrupted variant
    "labels": {"gamma": 4.0, "tv": 0.02, "iters": 200, "lr": 0.15, "keep_prior": 0.02, "mode": "linear"},
    # The app's SHIPPED configuration (Sources/CameraModel.swift TuningParams defaults as SRBridge maps them):
    # SNR auto-tune off, 16 px tiles, k_detail 0.17, k_denoise 1.0, k_stretch 3, k_shrink 1 (D_th / D_tr are not
    # exposed there, so the C++ defaults 0.76 / 1.12 apply). tile 0 would use the C++ SNR auto-tune instead.
    "harness": {"tile": 16, "k_detail": 0.17, "k_denoise": 1.0, "D_th": 0.76, "D_tr": 1.12, "k_stretch": 3.0,
                "k_shrink": 1.0},
    "workers": 2,
}

TRAIN_DEFAULTS = {
    "name": "m1",
    "seed": 0,
    "data": "data/d1",
    "model": {"ch": 16, "fusion": True, "dils": [1, 2, 4, 8], "up_ch": 8, "use_r0": False, "refine_r0": False},
    "crop": 128,             # raw px of the trained region per sample (<= data patch)
    "halo": 32,              # raw px of input context around it
    "batch": 4,
    "lr": 2.0e-3,
    "minutes": 40,
    "max_steps": 100000,
    "val_every": 250,
    "loss": {"align": 1.0, "merge": 1.0, "recon": 1.0, "align_sigma": 0.25, "gamma": 4.0, "keep_prior": 0.02,
             "w_harm": 3.0, "w_misalign": 3.0},
    "workers": 2,
    "threads": 4,
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path, defaults):
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    return _merge(defaults, user)


def load_data(path):
    return load(path, DATA_DEFAULTS)


def load_train(path):
    return load(path, TRAIN_DEFAULTS)


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def save(cfg, path):
    c = dict(cfg)
    c["_git"] = git_rev()
    c["_hash"] = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    with open(path, "w") as f:
        yaml.safe_dump(c, f, sort_keys=False)


def tool_path(*p):
    return os.path.join(TOOL, *p)
