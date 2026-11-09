"""MannNet checkpoint -> core/mann_weights.h (kMannTrained = true).
usage: python export_mann.py runs/mann1/best.pt path/to/core/mann_weights.h [--bias B] [--random SEED]
--bias adds B to the head bias (a validated operating-point shift); --random exports a random network (parity)."""
import argparse
import torch
from train_mann import MannNet

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("out")
ap.add_argument("--bias", type=float, default=0.0)
ap.add_argument("--random", type=int, default=None)
a = ap.parse_args()
m = MannNet()
if a.random is not None:
    torch.manual_seed(a.random)
    for c in list(m.convs) + [m.head]:
        torch.nn.init.normal_(c.weight, 0, 0.3)
        torch.nn.init.normal_(c.bias, 0, 0.1)
    torch.save(m.state_dict(), a.ckpt)
else:
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))


def arr(name, t):
    v = t.detach().reshape(-1).tolist()
    return f"inline const float {name}[{len(v)}] = {{" + ", ".join(f"{x:.9g}f" for x in v) + "};\n"


src = a.ckpt.replace("\\", "/")
out = ["#pragma once\n", f"// Misalignment-artifact NN weights (core/mann.h), exported by export_mann.py from {src}"
       + (f" (head bias {a.bias:+g})" if a.bias else "") + ".\n// Layout: PyTorch [cout][cin][3][3].\n",
       "namespace hhsr {\n", "constexpr bool kMannTrained = true;\n"]
for i, c in enumerate(m.convs):
    out.append(arr(f"kMannW{i}", c.weight))
    out.append(arr(f"kMannB{i}", c.bias))
out.append(arr("kMannWO", m.head.weight))
out.append(f"inline const float kMannBO = {m.head.bias.item() + a.bias:.9g}f;\n")
out.append("}  // namespace hhsr\n")
open(a.out, "w").write("".join(out))
print("wrote", a.out)
