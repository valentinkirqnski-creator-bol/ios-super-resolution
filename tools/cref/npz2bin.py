"""Rendered burst (.npz from real_protocol.py gen, RNN_LEAN=1) -> flat binary for cref_dataset.exe.
Layout: int32 N,H,W,cfa00,cfa01,cfa10,cfa11 | float32 alpha,beta,wb0,wb1,wb2 |
raw f32[N][H][W] | gt f32[N][2][H/2][W/2] (grey px, ref->frame k) | clean f32[N][H/2][W/2] | vis f32[N][H/2][W/2]
usage: python npz2bin.py in.npz out.bin"""
import sys
import numpy as np

d = np.load(sys.argv[1])
raw = d["raw"].astype(np.float32)
N, H, W = raw.shape
cfa = d["cfa"].astype(np.int32).reshape(-1)
wb = d["wb"].astype(np.float32)
with open(sys.argv[2], "wb") as f:
    np.array([N, H, W, *cfa], np.int32).tofile(f)
    np.array([float(d["alpha"]), float(d["beta"]), *wb], np.float32).tofile(f)
    raw.tofile(f)
    d["gt_flow"].astype(np.float32).tofile(f)
    d["clean_grey"].astype(np.float32).tofile(f)
    d["vis"].astype(np.float32).tofile(f)
