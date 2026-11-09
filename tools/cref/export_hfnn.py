"""Trained HfnnNet checkpoint -> core/hfnn_weights.h (kHfnnTrained = true).
usage: python export_hfnn.py runs/hfnn1/best.pt path/to/core/hfnn_weights.h"""
import sys
import torch
from train_hfnn import HfnnNet

m = HfnnNet()
m.load_state_dict(torch.load(sys.argv[1], map_location="cpu"))


def arr(name, t):
    v = t.detach().reshape(-1).tolist()
    return f"inline const float {name}[{len(v)}] = {{" + ", ".join(f"{x:.9g}f" for x in v) + "};\n"


out = ["#pragma once\n", "// HF-artifact NN weights (core/hfnn.h), exported by export_hfnn.py from "
       + sys.argv[1].replace("\\", "/") + ".\n// Layout: PyTorch [cout][cin][3][3].\n", "namespace hhsr {\n",
       "constexpr bool kHfnnTrained = true;\n"]
for i, c in enumerate(m.convs):
    out.append(arr(f"kHfnnW{i}", c.weight))
    out.append(arr(f"kHfnnB{i}", c.bias))
out.append(arr("kHfnnWO", m.head.weight))
out.append(f"constexpr float kHfnnBO = {m.head.bias.item():.9g}f;\n")
out.append("}  // namespace hhsr\n")
open(sys.argv[2], "w").write("".join(out))
print("wrote", sys.argv[2])
