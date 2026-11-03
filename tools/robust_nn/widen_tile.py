"""v34: widen a RobustNetTile checkpoint to the current feature count with zero weights on the new inputs (the net
starts exactly as before); the new channels' standardisation (mu / sd) comes from a pass over training batches.

    RNN_... python widen_tile.py runs/v33/model_s5484.pt runs/v34/init.pt c33O_train,c28p_train
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                   # noqa: E402
import train_pix as tp                         # noqa: E402
from rnn.features import NF                    # noqa: E402
from rnn.model import RobustNetTile            # noqa: E402

src, dst, sets = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
sd = torch.load(src, map_location="cpu")
nf0, hid, ctx = sd["c1.weight"].shape[1], sd["c1.weight"].shape[0], sd["k1.weight"].shape[0]
assert NF > nf0, (NF, nf0)
net = RobustNetTile(nf=NF, hid=hid, ctx=ctx)
new = net.state_dict()
for k, v in sd.items():
    if k == "c1.weight":
        new[k].zero_(); new[k][:, :nf0] = v
    elif k == "k1.weight":                       # [ctx, 2 nf, 1, 1]: avg block then max block
        new[k].zero_(); new[k][:, :nf0] = v[:, :nf0]; new[k][:, NF:NF + nf0] = v[:, nf0:]
    elif k in ("mu", "sd"):
        new[k][:nf0] = v
    else:
        new[k] = v
# standardisation of the new channels from data
acc, acc2, n = torch.zeros(NF - nf0, dtype=torch.float64), torch.zeros(NF - nf0, dtype=torch.float64), 0
for s in sets:
    ds = tp.Cache(f"data/{s}", 96, True)
    for i in range(min(len(ds), 12)):
        f = ds[i]["feats"][:, nf0:NF].double()
        acc += f.sum((0, 2, 3)); acc2 += (f * f).sum((0, 2, 3)); n += f[:, 0].numel()
mu = acc / n; sdv = (acc2 / n - mu * mu).clamp_min(1e-8).sqrt()
new["mu"][nf0:] = mu.float(); new["sd"][nf0:] = sdv.float()
# SEED=c (env): start the new inputs with the weights of existing input c (after standardisation both are "a shift
# in px", so the net responds to the predicted step from step 0 instead of learning the input from zero weight)
seed = os.environ.get("SEED")
if seed is not None:
    c = int(seed)
    for j in range(nf0, NF):
        new["c1.weight"][:, j] = new["c1.weight"][:, c]
        new["k1.weight"][:, j] = new["k1.weight"][:, c]
        new["k1.weight"][:, NF + j] = new["k1.weight"][:, NF + c]
    print(f"seeded inputs {nf0}..{NF - 1} with input {c}")
net.load_state_dict(new)
os.makedirs(os.path.dirname(dst), exist_ok=True)
torch.save(net.state_dict(), dst)
print(f"widened {nf0} -> {NF} inputs; new channel mu {mu.tolist()} sd {sdv.tolist()}; wrote {dst}")
