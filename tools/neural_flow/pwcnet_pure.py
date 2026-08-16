"""
PWCNet (sniklaus/pytorch-pwc architecture) with the CUDA/CuPy correlation
kernel replaced by a pure-PyTorch cost volume. Layer names match the
original exactly, so the pretrained state_dict loads unmodified.

Correlation math verified against external/pwcnet/correlation/correlation.py:
  - max displacement 4 in each axis -> 9x9 = 81 channels
  - channel index = (dy+4)*9 + (dx+4)   (outer loop dy, inner loop dx)
  - value = mean over channels of tenFirst[y,x,c] * tenSecond[y+dy,x+dx,c]
    (mean, not sum -- divided by sumelems = channel count, see
    kernel_Correlation_updateOutput: "total_sum / (float)sumelems"
    where sumelems = SIZE_3(rbot0) = channel count)
  - zero-padded border (the CUDA kernel pads rbot0/rbot1 by 4 px with zeros
    before the shift-and-dot; out-of-range displacements read those zeros)

Deployment note: every internal size (h, w at each pyramid level) is passed
down as a plain Python int rather than read from tensor .shape. Tracing an
intermediate tensor's .shape and feeding it into a tensor op (a division, a
linspace step count) captures a symbolic graph node even when the size is
fixed for a given deployment -- and coremltools' MIL conversion chokes on
several of those patterns (int-cast of a non-0-d array, reciprocal of an
int32 tensor). Since this model is being converted for one fixed resolution,
every level's size is known in closed form ahead of time (each level halves
the previous via stride-2 convs), so there's no loss of generality in
threading them through explicitly.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def correlation_pure(ten_first, ten_second, h, w, max_disp=4):
    pad = max_disp
    second_pad = F.pad(ten_second, [pad, pad, pad, pad])  # zero pad, matches CUDA kernel's zero border
    out = []
    for dy in range(-max_disp, max_disp + 1):
        for dx in range(-max_disp, max_disp + 1):
            shifted = second_pad[:, :, pad + dy:pad + dy + h, pad + dx:pad + dx + w]
            corr = (ten_first * shifted).mean(dim=1, keepdim=True)
            out.append(corr)
    return torch.cat(out, dim=1)  # (N, 81, H, W)


def backwarp(tenInput, tenFlow, h, w):
    # No module-level cache here (the original caches per-shape grids across
    # calls for training-time reuse) -- a Python-side cache that's populated
    # on first call and hit on later ones makes torch.jit.trace see two
    # different graphs on its two consistency-check invocations. Recomputing
    # is a one-time cost per traced size, not a hot loop, so it's fine to
    # drop for a fixed-shape deployed model.
    tenHor = torch.linspace(-1.0 + (1.0 / w), 1.0 - (1.0 / w), w).view(1, 1, 1, -1).expand(-1, -1, h, -1)
    tenVer = torch.linspace(-1.0 + (1.0 / h), 1.0 - (1.0 / h), h).view(1, 1, -1, 1).expand(-1, -1, -1, w)
    tenGrid = torch.cat([tenHor, tenVer], 1).to(tenInput.device)
    # .new_ones() isn't in coremltools' supported-op list ("PyTorch convert
    # function for op 'new_ones' not implemented"); torch.ones() with an
    # explicit batch of 1 (this model is converted for batch=1 only) covers
    # the same need via ops the converter does support.
    tenPartial = torch.ones((1, 1, h, w), dtype=tenFlow.dtype, device=tenFlow.device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((w - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], 1)
    tenInput = torch.cat([tenInput, tenPartial], 1)

    tenOutput = F.grid_sample(input=tenInput, grid=(tenGrid + tenFlow).permute(0, 2, 3, 1),
                              mode='bilinear', padding_mode='zeros', align_corners=False)

    tenMask = tenOutput[:, -1:, :, :]
    tenMask = (tenMask > 0.999).float()

    return tenOutput[:, :-1, :, :].contiguous() * tenMask.contiguous()


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Extractor(nn.Module):
            def __init__(self):
                super(Extractor, self).__init__()
                self.netOne = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netTwo = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netThr = nn.Sequential(
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netFou = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netFiv = nn.Sequential(
                    nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                self.netSix = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)
                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]

        class Decoder(nn.Module):
            def __init__(self, intLevel):
                super(Decoder, self).__init__()
                intPrevious = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 1]
                intCurrent = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 0]

                if intLevel < 6: self.netUpflow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.netUpfeat = nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

                self.netOne = nn.Sequential(nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netTwo = nn.Sequential(nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netThr = nn.Sequential(nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFou = nn.Sequential(nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netFiv = nn.Sequential(nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(inplace=False, negative_slope=0.1))
                self.netSix = nn.Sequential(nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1))

            def forward(self, tenFirst, tenSecond, objPrevious, h, w):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenVolume = F.leaky_relu(input=correlation_pure(tenFirst, tenSecond, h, w), negative_slope=0.1, inplace=False)
                    tenFeat = torch.cat([tenVolume], 1)
                else:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat = self.netUpfeat(objPrevious['tenFeat'])
                    tenVolume = F.leaky_relu(input=correlation_pure(tenFirst, backwarp(tenInput=tenSecond, tenFlow=tenFlow * self.fltBackwarp, h=h, w=w), h, w), negative_slope=0.1, inplace=False)
                    tenFeat = torch.cat([tenVolume, tenFirst, tenFlow, tenFeat], 1)

                tenFeat = torch.cat([self.netOne(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netTwo(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netThr(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFou(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFiv(tenFeat), tenFeat], 1)

                tenFlow = self.netSix(tenFeat)
                return {'tenFlow': tenFlow, 'tenFeat': tenFeat}

        class Refiner(nn.Module):
            def __init__(self):
                super(Refiner, self).__init__()
                self.netMain = nn.Sequential(
                    nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )

            def forward(self, tenInput):
                return self.netMain(tenInput)

        self.netExtractor = Extractor()
        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)
        self.netRefiner = Refiner()

    def forward(self, tenFirst, tenSecond, sizes):
        # sizes: [(h,w) for netOne, netTwo, netThr, netFou, netFiv, netSix],
        # i.e. input_h//2, //4, //8, //16, //32, //64 -- plain Python ints,
        # see module docstring for why these can't be read from .shape here.
        tenFirst = self.netExtractor(tenFirst)
        tenSecond = self.netExtractor(tenSecond)

        h6, w6 = sizes[5]
        h5, w5 = sizes[4]
        h4, w4 = sizes[3]
        h3, w3 = sizes[2]
        h2, w2 = sizes[1]

        objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None, h6, w6)
        objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate, h5, w5)
        objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate, h4, w4)
        objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate, h3, w3)
        objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate, h2, w2)

        return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])


class PWCNet(nn.Module):
    def __init__(self, load_pretrained=True, weights_path=None, rgb2bgr=False):
        super(PWCNet, self).__init__()
        self.net = Network()
        self.rgb2bgr = rgb2bgr

        if load_pretrained:
            if weights_path is None:
                raise Exception
            weights_dict = torch.load(weights_path, map_location='cpu')
            self.net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in weights_dict.items()})

    def forward(self, source_img, target_img, int_height=None, int_width=None):
        # int_height/int_width: plain Python ints, fixed at conversion time.
        # The original derives these from .shape[-2:] at call time, which is
        # necessary for a general-purpose library entry point but traces into
        # symbolic-int ops that coremltools' MIL conversion chokes on. A
        # model built for one deployment resolution doesn't need that
        # generality -- passing the sizes in as constants sidesteps the
        # tensor-shape arithmetic entirely and traces to plain constants.
        if int_height is None:
            int_height = source_img.shape[-2]
        if int_width is None:
            int_width = source_img.shape[-1]

        source_img = source_img.view(-1, 3, int_height, int_width)
        target_img = target_img.view(-1, 3, int_height, int_width)

        if self.rgb2bgr:
            source_img = source_img[:, [2, 1, 0]].contiguous()
            target_img = target_img[:, [2, 1, 0]].contiguous()

        int_preprocessed_width = int(math.floor(math.ceil(int_width / 64.0) * 64.0))
        int_preprocessed_height = int(math.floor(math.ceil(int_height / 64.0) * 64.0))

        source_img_re = F.interpolate(input=source_img, size=(int_preprocessed_height, int_preprocessed_width), mode='bilinear', align_corners=False)
        target_img_re = F.interpolate(input=target_img, size=(int_preprocessed_height, int_preprocessed_width), mode='bilinear', align_corners=False)

        # Each Extractor stage halves via a stride-2 conv; int_preprocessed_*
        # is already a multiple of 64 so every level below divides evenly.
        sizes = [(int_preprocessed_height // d, int_preprocessed_width // d) for d in (2, 4, 8, 16, 32, 64)]

        flow = self.net(target_img_re, source_img_re, sizes)
        flow = 20.0 * F.interpolate(input=flow, size=(int_height, int_width), mode='bilinear', align_corners=False)

        scale_factor_x = float(int_width) / float(int_preprocessed_width)
        scale_factor_y = float(int_height) / float(int_preprocessed_height)
        flow = torch.stack((flow[:, 0] * scale_factor_x, flow[:, 1] * scale_factor_y), dim=1)

        return flow
