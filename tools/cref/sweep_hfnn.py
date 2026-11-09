import torch, numpy as np
from crops import load_dir
import train_hfnn as T
va = load_dir(r"data\val")
m = T.HfnnNet(); m.load_state_dict(torch.load(r"runs\hfnn1\best.pt"))
b0 = m.head.bias.item()
print("bias offset | alg6 catch false | gcam catch false   (HF-pattern px; mask alone: alg6 0.211/0.078, gcam 0.594/0.391)")
for off in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    with torch.no_grad(): m.head.bias.fill_(b0 + off)
    s, r = T.evaluate(m, va, 0.35)
    print("%+.2f | %.3f %.3f | %.3f %.3f | score %.3f" % (off, r[0]['catch'], r[0]['false'], r[1]['catch'], r[1]['false'], s))
