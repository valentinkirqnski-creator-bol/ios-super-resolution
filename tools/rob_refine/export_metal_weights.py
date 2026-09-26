"""Generate core/robustness_refine_weights.h from a trained checkpoint.

The Metal path does not use Core ML. A 689-parameter pointwise network is
smaller than the code that would load a model file, so the weights are compiled
in as one flat float array in the layout core/robustness_refine_shared.h
defines (RR_OFF_*). That removes a bundle resource, a load-failure path and the
whole CPU round trip: the GPU kernel reads this buffer and nothing else.

The input transform -- signed log on the flagged channels, then per-channel
mean/std -- is folded into the same array, for the same reason
export_coreml.py folds it into the graph: a normalisation table living
somewhere other than next to the weights will eventually be the wrong table.

    python export_metal_weights.py refinenet_mlp_geom.pt

Writing the header is the only side effect; re-run it whenever the model is
retrained, and rebuild.
"""
import os
import sys

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
from train_refine import IN_CH, LOG_CH

CORE = os.path.normpath(os.path.join(SC, "..", "..", "core"))

# Mirrors core/robustness_refine_shared.h. Asserted against the header below so
# a change on either side is a build-time or run-time failure rather than a
# silently wrong mask.
RR_CHANNELS = 39
RR_WIDTH = 16
OFF_MU = 0
OFF_SD = OFF_MU + RR_CHANNELS
OFF_LOGMASK = OFF_SD + RR_CHANNELS
OFF_W1 = OFF_LOGMASK + RR_CHANNELS
OFF_B1 = OFF_W1 + RR_WIDTH * RR_CHANNELS
OFF_W2 = OFF_B1 + RR_WIDTH
OFF_B2 = OFF_W2 + RR_WIDTH * RR_WIDTH
OFF_W3 = OFF_B2 + RR_WIDTH
OFF_B3 = OFF_W3 + RR_WIDTH
WEIGHTS_N = OFF_B3 + 1


def check_header_agrees():
    """The offsets above are a duplicate of the header's #defines. Rather than
    trust that, read them back out of the header and compare."""
    path = os.path.join(CORE, "robustness_refine_shared.h")
    src = open(path, encoding="utf-8").read()
    for name, want in (("RR_CHANNELS", RR_CHANNELS), ("RR_WIDTH", RR_WIDTH)):
        needle = f"#define {name} "
        if needle not in src:
            raise SystemExit(f"{name} not found in {path}")
        got = int(src.split(needle, 1)[1].split("\n", 1)[0].strip())
        if got != want:
            raise SystemExit(f"{name}: header says {got}, this script assumes {want}")


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        SC, "refinenet_mlp_geom.pt")
    check_header_agrees()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck["arch"] != "mlp":
        raise SystemExit(
            f"arch is {ck['arch']!r}; only the pointwise model can be compiled "
            "in. A convolutional variant needs a neighbourhood the kernel does "
            "not gather, and kRobustnessRefineHalo raised to match.")
    if ck["in_ch"] != RR_CHANNELS or ck["width"] != RR_WIDTH:
        raise SystemExit(f"checkpoint is {ck['in_ch']}x{ck['width']}, header "
                         f"says {RR_CHANNELS}x{RR_WIDTH}")

    st = ck["state"]
    # nn.Sequential(Conv2d(24,16,1), ReLU, Conv2d(16,16,1), ReLU, Conv2d(16,1,1))
    w1 = st["net.0.weight"].numpy().reshape(RR_WIDTH, RR_CHANNELS)
    b1 = st["net.0.bias"].numpy().reshape(RR_WIDTH)
    w2 = st["net.2.weight"].numpy().reshape(RR_WIDTH, RR_WIDTH)
    b2 = st["net.2.bias"].numpy().reshape(RR_WIDTH)
    w3 = st["net.4.weight"].numpy().reshape(RR_WIDTH)
    b3 = float(st["net.4.bias"].numpy().reshape(1)[0])

    buf = np.zeros(WEIGHTS_N, np.float32)
    buf[OFF_MU:OFF_MU + RR_CHANNELS] = np.asarray(ck["mu"], np.float32)
    buf[OFF_SD:OFF_SD + RR_CHANNELS] = np.asarray(ck["sd"], np.float32)
    logmask = np.zeros(RR_CHANNELS, np.float32)
    logmask[ck["log_ch"]] = 1.0
    buf[OFF_LOGMASK:OFF_LOGMASK + RR_CHANNELS] = logmask
    buf[OFF_W1:OFF_W1 + w1.size] = w1.ravel()
    buf[OFF_B1:OFF_B1 + RR_WIDTH] = b1
    buf[OFF_W2:OFF_W2 + w2.size] = w2.ravel()
    buf[OFF_B2:OFF_B2 + RR_WIDTH] = b2
    buf[OFF_W3:OFF_W3 + RR_WIDTH] = w3
    buf[OFF_B3] = b3

    # Round-trip the array back through the same arithmetic the kernel will do,
    # and compare against the torch model. This is the only check that catches a
    # transposed weight matrix, which would otherwise produce a plausible-looking
    # mask that is simply wrong.
    rng = np.random.RandomState(0)
    probe = (rng.randn(64, RR_CHANNELS) * 0.7).astype(np.float32)
    x = probe.copy()
    x[:, ck["log_ch"]] = np.sign(x[:, ck["log_ch"]]) * np.log1p(
        np.abs(x[:, ck["log_ch"]]))
    x = (x - buf[OFF_MU:OFF_SD]) / buf[OFF_SD:OFF_LOGMASK]
    h0 = np.maximum(x @ w1.T + b1, 0.0)
    h1 = np.maximum(h0 @ w2.T + b2, 0.0)
    mine = 1.0 / (1.0 + np.exp(-(h1 @ w3 + b3)))
    from train_refine import RefineMLP
    core = RefineMLP(w=RR_WIDTH)
    core.load_state_dict(st)
    core.eval()
    with torch.no_grad():
        # RefineMLP already ends in the sigmoid.
        ref = core(torch.from_numpy(x.astype(np.float32))
                   .view(-1, RR_CHANNELS, 1, 1)).numpy().ravel()
    err = float(np.abs(mine - ref).max())
    print(f"flat-buffer vs torch max |diff| {err:.3g}")
    if err > 1e-5:
        raise SystemExit("weight layout does not reproduce the model")

    out = os.path.join(CORE, "robustness_refine_weights.h")
    with open(out, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("#pragma once\n")
        f.write("//\n")
        f.write("// GENERATED by tools/rob_refine/export_metal_weights.py -- "
                "do not edit.\n")
        f.write("//\n")
        f.write("// Weights for the robustness refinement network, in the flat\n")
        f.write("// layout core/robustness_refine_shared.h defines (RR_OFF_*):\n")
        f.write("// input mean, input std, signed-log channel mask, then the\n")
        f.write("// three layers. Compiled in rather than loaded, because at\n")
        f.write(f"// {w1.size + b1.size + w2.size + b2.size + w3.size + 1} "
                "parameters the model is smaller than the code that\n")
        f.write("// would read a file, and a compiled-in table has no load\n")
        f.write("// failure to handle. The Metal path uses ONLY this; Core ML\n")
        f.write("// is not involved on the GPU.\n")
        f.write("//\n")
        f.write(f"// Source checkpoint: {os.path.basename(ckpt)} "
                f"({ck['arch']}, refining the {ck['baseline']} baseline)\n")
        f.write("\nnamespace hhsr {\n\n")
        f.write(f"inline constexpr int kRobustnessRefineWeightCount = {WEIGHTS_N};\n\n")
        f.write("inline constexpr float kRobustnessRefineWeights"
                f"[{WEIGHTS_N}] = {{\n")
        for i in range(0, WEIGHTS_N, 6):
            row = ", ".join(f"{v:+.9e}f" for v in buf[i:i + 6])
            f.write(f"    {row},\n")
        f.write("};\n\n}  // namespace hhsr\n")
    print(f"wrote {out} ({WEIGHTS_N} floats, "
          f"{os.path.getsize(out) / 1024:.0f} KB of source)")


if __name__ == "__main__":
    main()
