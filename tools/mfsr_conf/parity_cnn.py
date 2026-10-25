"""Parity: the device-side ConfNet code (core/conf_nn_shared.h, compiled as C++ by cpp/conf_parity.cpp) against the
PyTorch model and input pipeline, on a stored sample, for every architecture variant.

    python parity_cnn.py [--model runs/<x>/best.pt]      (default: random weights for each variant)

Checks: the 18 guide inputs computed from greys / flows (features + pixel channels), the 3 raw inputs, the network
logits (standardisation folding, dilated convs, fusion, head, raw refinement) and R. Prints PARITY OK / FAIL.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np            # noqa: E402
import torch                  # noqa: E402

from mfsr import merge as mg, model as M                                         # noqa: E402
from mfsr.inputs import frame_inputs, guide_flow, guide_grey, grey_noise         # noqa: E402
from rnn.common import bilinear_clamp                                            # noqa: E402
import export_weights as EW                                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))


def build(header, exe):
    F = ["-O2", "-std=gnu++17", f"-I{ROOT}/core", f"-DCNN_WEIGHTS_HEADER=\"{header}\"", "-static"]
    subprocess.run(["g++", *F, os.path.join(HERE, "cpp", "conf_parity.cpp"), "-o", exe], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--sample", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    fn = a.sample or sorted(glob.glob(os.path.join(HERE, "data", "d1", "val", "*.npz")))[0]
    z = np.load(fn)
    meta = json.loads(str(z["meta"]))
    cfa = tuple(tuple(r) for r in meta["cfa"])
    wb = np.array(meta["prewhite"], np.float32)
    na, nb, tile = meta["noise_a"], meta["noise_b"], meta["tile"]
    v = 1
    raw = torch.from_numpy(z["raw"].astype(np.float32))
    N, S, _ = raw.shape
    K, G = N - 1, S // 2
    flows, Hs, hh = torch.from_numpy(z[f"flow{v}"]), torch.from_numpy(z[f"H{v}"]), z[f"hash{v}"]
    r0g = torch.from_numpy(z[f"r0{v}"].astype(np.float32))
    greyr, grey = torch.from_numpy(z["grey_ref"].astype(np.float32)), torch.from_numpy(z["grey"].astype(np.float32))
    a_g, b_g = grey_noise(na, nb)
    gg = guide_grey(raw, wb, cfa)
    fw, ff = [], []
    for k in range(K):
        f_, ff_ = guide_flow(flows[k], Hs[k], bool(hh[k] > 0.5), tile, G)
        fw.append(f_), ff.append(ff_)
    fw, ff = torch.stack(fw), torch.stack(ff)
    yy, xx = torch.meshgrid(torch.arange(S, dtype=torch.float32), torch.arange(S, dtype=torch.float32), indexing="ij")
    jf = []
    for k in range(K):
        my, mx = mg.tile_correspondence(yy, xx, flows[k], Hs[k], bool(hh[k] > 0.5), tile)
        jf.append(bilinear_clamp(grey[k][None], mx[None], my[None])[0])
    jf = torch.stack(jf)
    r0raw = r0g.repeat_interleave(2, -2).repeat_interleave(2, -1)
    if a.model:
        ck = torch.load(a.model, map_location="cpu", weights_only=False)
        variants = [(ck["cfg"]["model"], ck["state"])]
    else:
        variants = [(dict(ch=16, fusion=True, dils=[1, 2, 4, 8], up_ch=8, use_r0=False, refine_r0=False), None),
                    (dict(ch=16, fusion=False, dils=[1, 2, 4, 8], up_ch=0, use_r0=False, refine_r0=False), None),
                    (dict(ch=12, fusion=True, dils=[1, 1, 2, 2], up_ch=6, use_r0=True, refine_r0=True), None)]
    ok_all = True
    for mc, state in variants:
        use_r0 = bool(mc.get("use_r0") or mc.get("refine_r0"))
        xg, xr = frame_inputs(raw, greyr, grey, r0g, flows, Hs, hh, tile, na, nb, wb, use_r0, cfa)
        net = M.build(mc)
        if state is not None:
            net.load_state_dict(state)
        else:
            with torch.no_grad():
                net.mu_g.copy_(torch.randn_like(net.mu_g) * 0.5)
                net.sd_g.copy_(torch.rand_like(net.sd_g) + 0.5)
                net.mu_r.copy_(torch.randn_like(net.mu_r) * 0.5)
                net.sd_r.copy_(torch.rand_like(net.sd_r) + 0.5)
                if net.up_ch > 0:
                    net.u2.weight.normal_(0, 0.2)
                    net.u2.bias.normal_(0, 0.1)
        net.eval()
        with torch.no_grad():
            lg = net(xg[None], xr[None], torch.ones(1, K), r0raw[None] if mc.get("refine_r0") else None)[0]
            Rpy = torch.sigmoid(lg[:, 1])
            if mc.get("refine_r0"):        # the C++ adds logit(R0) inside cnn_weight, not to the reported logit
                lg = lg.clone()
                lg[:, 1] -= torch.logit(r0raw.clamp(1e-3, 1 - 1e-3))
        with tempfile.TemporaryDirectory() as d:
            hdr, exe = os.path.join(d, "w.h"), os.path.join(d, "p.exe")
            EW.export(net, mc, hdr, tag="parity")
            build(hdr.replace("\\", "/"), exe)
            pi, po = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
            with open(pi, "wb") as fh:
                np.array([K, G, G, S, S, max(1, tile // 2)], np.int32).tofile(fh)
                np.array([a_g, b_g, 0.9], np.float32).tofile(fh)
                for t in (gg[0], gg[1:], fw.permute(0, 2, 3, 1), ff.permute(0, 2, 3, 1), r0g, xg, greyr, jf, r0raw):
                    np.ascontiguousarray(t.numpy(), np.float32).tofile(fh)
            subprocess.run([exe, pi, po], check=True)
            o = np.fromfile(po, np.float32)
        n1, n2, n3 = K * xg.shape[1] * G * G, K * xr.shape[1] * S * S, K * 2 * S * S
        xg_c = torch.from_numpy(o[:n1]).view_as(xg)
        xr_c = torch.from_numpy(o[n1:n1 + n2]).view_as(xr)
        lg_c = torch.from_numpy(o[n1 + n2:n1 + n2 + n3]).view_as(lg)
        R_c = torch.from_numpy(o[n1 + n2 + n3:]).view_as(Rpy)
        e = dict(inputs_guide=(xg_c - xg).abs().max().item(), inputs_raw=(xr_c - xr).abs().max().item(),
                 logits=(lg_c - lg).abs().max().item(), R=(R_c - Rpy).abs().max().item())
        ok = e["inputs_guide"] < 1e-3 and e["inputs_raw"] < 1e-3 and e["logits"] < 2e-3 and e["R"] < 5e-4
        ok_all &= ok
        print(f"{'OK  ' if ok else 'FAIL'} {mc}: max abs diff " + ", ".join(f"{k} {v_:.2e}" for k, v_ in e.items()))
    print("PARITY OK" if ok_all else "PARITY FAIL")


if __name__ == "__main__":
    main()
