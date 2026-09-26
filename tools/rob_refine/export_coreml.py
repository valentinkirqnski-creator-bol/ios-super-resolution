"""Export a trained refinement network for the app.

The exported graph carries the input transform -- the signed log on the
heavy-tailed channels, then the per-channel mean/std -- so the C++ side hands
over the raw feature planes build_robustness_refine_features produced and
there is no constant table living outside the weights to drift out of step
with them. That matters more here than usual: the features are physical
quantities in raw pixels, and a normalisation mismatch would not crash, it
would quietly bias the mask.

Input  "features": (1, kRobustnessRefineChannels, H, W) float32
Output "keep":     (1, 1, H, W) float32 in [0,1], q_keep

H and W are flexible so one model serves any strip height the caller picks.

    python export_coreml.py refinenet_mlp_geom.pt RobustnessRefineNet.mlpackage
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
from train_refine import RefineMLP, RefineCNN, IN_CH


class Wrapped(nn.Module):
    """The trained network with its input transform baked in front."""

    def __init__(self, core, mu, sd, log_ch):
        super().__init__()
        self.core = core
        self.register_buffer(
            "mu", torch.from_numpy(np.asarray(mu, np.float32)).view(1, -1, 1, 1))
        self.register_buffer(
            "sd", torch.from_numpy(np.asarray(sd, np.float32)).view(1, -1, 1, 1))
        # A 0/1 selector rather than fancy indexing: index assignment does not
        # trace cleanly, while a mask multiply is two elementwise ops every
        # backend handles.
        m = np.zeros((1, IN_CH, 1, 1), np.float32)
        m[0, log_ch, 0, 0] = 1.0
        self.register_buffer("logmask", torch.from_numpy(m))

    def forward(self, features):
        lg = torch.sign(features) * torch.log1p(torch.abs(features))
        x = self.logmask * lg + (1.0 - self.logmask) * features
        return self.core((x - self.mu) / self.sd)


def check_equivalent(model, core, ck):
    """The baked-in transform must reproduce exactly what training applied.
    A mismatch here does not crash on device, it biases the mask -- so it is
    checked rather than assumed."""
    rng = np.random.RandomState(0)
    probe = (rng.randn(1, IN_CH, 8, 8) * 0.3).astype(np.float32)
    with torch.no_grad():
        got = model(torch.from_numpy(probe)).numpy()
    manual = probe.copy()
    ch = ck["log_ch"]
    manual[:, ch] = np.sign(probe[:, ch]) * np.log1p(np.abs(probe[:, ch]))
    manual = (manual - np.asarray(ck["mu"], np.float32).reshape(1, -1, 1, 1))
    manual = manual / np.asarray(ck["sd"], np.float32).reshape(1, -1, 1, 1)
    with torch.no_grad():
        want = core(torch.from_numpy(manual.astype(np.float32))).numpy()
    err = float(np.abs(got - want).max())
    print(f"wrapped-vs-manual max |diff| {err:.3g}")
    if err >= 1e-5:
        raise SystemExit("the baked-in transform does not match training")


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        SC, "refinenet_mlp_geom.pt")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        SC, "RobustnessRefineNet.mlpackage")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    core = RefineMLP(w=ck["width"]) if ck["arch"] == "mlp" else RefineCNN(w=ck["width"])
    core.load_state_dict(ck["state"])
    core.eval()
    model = Wrapped(core, ck["mu"], ck["sd"], ck["log_ch"]).eval()
    check_equivalent(model, core, ck)

    import coremltools as ct

    traced = torch.jit.trace(model, torch.zeros(1, IN_CH, 64, 64))
    shape = ct.Shape(shape=(1, IN_CH, ct.RangeDim(8, 4096), ct.RangeDim(8, 8192)))
    inputs = [ct.TensorType(name="features", shape=shape)]
    try:
        m = ct.convert(traced, convert_to="mlprogram", inputs=inputs,
                       outputs=[ct.TensorType(name="keep")],
                       minimum_deployment_target=ct.target.iOS15)
    except Exception as e:
        # coremltools ships its BlobWriter extension only for macOS and Linux,
        # so an mlprogram weight blob cannot be written on Windows. The older
        # neuralnetwork format needs no extension and loads identically on
        # device; re-run this on a Mac for the smaller fp16 mlpackage.
        print(f"mlprogram export unavailable ({type(e).__name__}: {e}); "
              "falling back to neuralnetwork")
        m = ct.convert(traced, convert_to="neuralnetwork", inputs=inputs)
        if out.endswith(".mlpackage"):
            out = out[: -len(".mlpackage")] + ".mlmodel"

    # robustness_nn.mm asks for the output by name, and the neuralnetwork
    # converter names it after whatever the traced graph's last node was.
    # Renaming here keeps that contract in one place instead of teaching the
    # Objective-C side to guess.
    spec = m.get_spec()
    produced = spec.description.output[0].name
    if produced != "keep":
        ct.utils.rename_feature(spec, produced, "keep")
        m = ct.models.MLModel(spec)
        print(f"renamed output '{produced}' -> 'keep'")

    m.short_description = ("Conservative refinement of the Wronski robustness "
                           "mask: predicts q_keep, applied by the pipeline as "
                           "R * (1 - kappa * (1 - q_keep)).")
    m.save(out)
    n = sum(p.numel() for p in core.parameters())
    print(f"saved {out}")
    print(f"{ck['arch']} / {ck['baseline']} baseline, {n} parameters, "
          f"receptive field {'1x1' if ck['arch'] == 'mlp' else '5x5'}")
    if ck["arch"] != "mlp":
        print("NOTE: raise kRobustnessRefineHalo to 2 in core/types.h before "
              "shipping a cnn model, or the inference strips will seam.")


if __name__ == "__main__":
    main()
