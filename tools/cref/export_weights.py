"""Trained CrefNet checkpoint -> core/cref_weights.h (kCrefTrained = true).
usage: python export_weights.py runs/cref1/best.pt path/to/core/cref_weights.h"""
import sys
import torch
from train_cref import CrefNet

m = CrefNet()
m.load_state_dict(torch.load(sys.argv[1], map_location="cpu"))


def arr(name, t):
    v = t.detach().reshape(-1).tolist()
    return f"inline const float {name}[{len(v)}] = {{" + ", ".join(f"{x:.9g}f" for x in v) + "};\n"


out = ["#pragma once\n", "// Consensus refiner weights (core/cref.h), exported by export_weights.py from "
       + sys.argv[1].replace("\\", "/") + ".\n// Layout: PyTorch [cout][cin][3][3].\n", "namespace hhsr {\n",
       "constexpr bool kCrefTrained = true;\n"]
for i, c in enumerate(m.convs):
    out.append(arr(f"kCrefW{i}", c.weight))
    out.append(arr(f"kCrefB{i}", c.bias))
out.append(arr("kCrefWO", m.head.weight))
out.append(f"constexpr float kCrefBO = {m.head.bias.item():.9g}f;\n")
out.append("}  // namespace hhsr\n")
open(sys.argv[2], "w").write("".join(out))
print("wrote", sys.argv[2])
