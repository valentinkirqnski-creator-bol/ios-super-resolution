# fusion_nn: learned per-frame fusion weights for the Wronski merge (prototype)

This network replaces the analytic robustness (R) of the Wronski et al. path
(core/robustness.cpp + core/merge.cpp). It keeps the app's flow and kernel
regression. It is written from scratch: no existing network, weights or burst
generator is reused.

## Fusion

For each output pixel, `core/merge.cpp` computes per frame f and colour c a
numerator A_fc and a denominator B_fc. These come from the 3x3 raw samples
around the flow-warped position, weighted by the anisotropic kernel. The frame
weight is constant over those samples, so for any weights w:

    out_c = Σ_f w_f A_fc / Σ_f w_f B_fc

- **Analytic:** w = (1, R_1, …, R_N-1).
- **Network:** w_f = softmax over frames of a per-frame logit. These are
  normalised contribution weights, and the reference frame is included.
- **Parity:** `fusion.merge_terms` ports accumulate_ref / accumulate_comp
  line by line. Against the app's merged image the mean |diff| is ~1e-6. The
  largest differences, 2e-4 to 2e-2 on a few pixels, come from the f16-stored
  kernel covariances.
- **Kernel regression is kept:** the network does not average per-frame
  interpolations. It weights each frame's own kernel-regression terms.
- **On the device:** the softmax denominator cancels in the ratio, so the merge
  streams frame by frame as today. Each comparison frame's weight relative to
  the reference is e^(l_f - l_ref), so no stack of N weight maps is needed.

## Network (`fusion.FusionNet`)

- Runs per frame with shared weights, at guide resolution (half raw), on the
  same grid as R.
- **Inputs (8 channels):** is-reference; reference brightness; (warped frame -
  reference); (warped frame - burst mean of warped frames). The last two are
  in noise std via GAT and the DNG noise profile. Warping uses the app's tile
  flow.
- **Layers:** 3x3 conv 8→12, then 4 residual depthwise-separable blocks
  (3x3 depthwise, dilations 1/2/4/8, then 1x1 12→12, ReLU), then 1x1 → logit.
  - The receptive field is ±16 guide px.
  - 1,993 parameters (4 KB fp16).
  - 1,884 MAC per guide pixel.
- **Initialisation:** a zero head, so it starts as "all frames equal".

## Data (`gen.py`)

- **Sources:** APC_1294 (burst12 frame 0) and APC_1186 (burst9 frame 0).
- **Clean scene:** each 2x2 Bayer quad becomes one linear RGB pixel, giving
  2016x1512 per frame.
- **Bursts:** 8 frames of 256x256 raw each, with:
  - camera shift (up to ±4 px), rotation, scale and shear;
  - parallax: an inverse-depth map times the camera translation;
  - 0–2 moving scene patches (occlusion);
  - exposure variation with a ±2 % compensation residual, clipping;
  - Poisson-Gaussian noise from the DNG profile × ISO factor 1..32;
  - 12-bit codes.
- **App stages:** every burst then goes through the app's own alignment,
  kernel estimation, analytic robustness and merge (`cpp/wronski_dump.cpp`,
  built from core/).
- **Split:** the right 20 % of each scene is validation. A 192 px guard band
  keeps every training crop, and every pixel it samples, out of it.
- **Size:** 1,000 training + 100 validation bursts, ~2.3 MB each (.npz,
  compressed).

## Train / evaluate

    sh build.sh                                      # wronski_dump (MinGW, the app's core/)
    python gen.py --scenes SCENES --out D/val   --split val   --n 100  --seed0 1000000
    python gen.py --scenes SCENES --out D/train --split train --n 1000 --seed0 0
    python check.py D/train/train_0000000.npz        # merge parity + flow vs truth
    python train.py --data D/train --val D/val --out D/runs/r1
    python eval.py --val D/val --model D/runs/r1/best.pt --out D/runs/r1/eval.json

## Limits

- Two daylight scenes and simulated motion. This is a proof of concept on
  these textures, not evidence of generalisation.
- The real burst12 and burst9 share these scenes, so a visual check on them is
  not an independent test.
- Device timing and memory must be measured on the iPhone. The figures in
  eval.py are MAC counts, buffer sizes and PC timings.
