"""v37: the shipped v21 base net plus a zero-output tile-aligned wide view (RobustNetTile, 14 inputs): starts
exactly as v21 and gains the ~2x2-tile view in training.    python widen_v21_tile.py runs/v21/model.pt runs/v37/init.pt"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                   # noqa: E402
from rnn.model import RobustNetTile            # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu")
net = RobustNetTile(nf=sd["c1.weight"].shape[1], hid=sd["c1.weight"].shape[0], ctx=8)
missing = net.load_state_dict(sd, strict=False).missing_keys
assert all(k.startswith("k") or k == "tile_model" for k in missing), missing
os.makedirs(os.path.dirname(dst), exist_ok=True)
torch.save(net.state_dict(), dst)
print(f"wrote {dst}: {sd['c1.weight'].shape[1]} inputs, tile branch at zero")
