"""Random-weights parity for the HF-artifact NN: 'make' writes a random HfnnNet + its header;
'check' runs PyTorch on the dumped full-frame features, applies the app's HF blend and compares
with the CPU hfnn_refine output.  usage: python hfnn_parity.py make outdir | check outdir H W"""
import os
import subprocess
import sys
import numpy as np
import torch
from train_hfnn import HfnnNet, blend_w

mode, d = sys.argv[1], sys.argv[2]
if mode == "make":
    torch.manual_seed(2)
    m = HfnnNet()
    for c in m.convs:
        torch.nn.init.normal_(c.weight, 0, 0.3); torch.nn.init.normal_(c.bias, 0, 0.1)
    torch.nn.init.normal_(m.head.weight, 0, 0.5); torch.nn.init.normal_(m.head.bias, 0, 0.2)
    os.makedirs(os.path.join(d, "inc"), exist_ok=True)
    torch.save(m.state_dict(), os.path.join(d, "hf_rand.pt"))
    subprocess.check_call([sys.executable, "-B", "export_hfnn.py", os.path.join(d, "hf_rand.pt"),
                           os.path.join(d, "inc", "hfnn_weights.h")])
else:
    H, W = int(sys.argv[3]), int(sys.argv[4])
    m = HfnnNet()
    m.load_state_dict(torch.load(os.path.join(d, "hf_rand.pt")))
    F = np.fromfile(os.path.join(d, "hf_feats.f32"), np.float32).reshape(1, 6, H, W)
    x = torch.from_numpy(F)
    with torch.no_grad():
        rn = m(x)
        R0 = x[:, :1]
        R = (R0 + blend_w(x[:, 5:6], 0.35) * (rn - R0))[0, 0].numpy()
    rc = np.fromfile(os.path.join(d, "hf_rcpu.f32"), np.float32).reshape(H, W)
    d_ = np.abs(R - rc)
    print("torch vs CPU max |diff| %.3g (interior %.3g); pixels changed by the NN: %.3f; HF px: %.3f"
          % (d_.max(), d_[8:-8, 8:-8].max(), (np.abs(R - F[0, 0]) > 1e-6).mean(), (F[0, 5] > 0.25).mean()))
