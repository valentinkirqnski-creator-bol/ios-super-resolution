import sys
import torch
import torch.nn as nn
import coremltools as ct
from pwcnet_pure import PWCNet

H, W = 1512, 2016  # guide resolution, matches the app's existing robustness/align grid


class PWCNetCoreML(nn.Module):
    """Fixed-size wrapper: two RGB guide frames (0..1, NCHW) -> dense flow (1,2,H,W)."""
    def __init__(self, weights_path):
        super().__init__()
        self.pwc = PWCNet(load_pretrained=True, weights_path=weights_path)
        self.pwc.eval()

    def forward(self, ref, comp):
        return self.pwc(ref, comp, int_height=H, int_width=W)


def main():
    m = PWCNetCoreML('pretrained_networks/pwcnet-network-default.pth')
    m.eval()

    ref = torch.rand(1, 3, H, W)
    comp = torch.rand(1, 3, H, W)

    print('tracing...')
    with torch.no_grad():
        traced = torch.jit.trace(m, (ref, comp))
    print('trace OK')

    print('converting to Core ML (mlprogram)...')
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name='ref_frame', shape=ref.shape),
            ct.TensorType(name='comp_frame', shape=comp.shape),
        ],
        outputs=[ct.TensorType(name='flow')],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to='mlprogram',
    )
    out_path = 'PWCNetFlow.mlpackage'
    mlmodel.save(out_path)
    print('saved', out_path)


if __name__ == '__main__':
    main()
