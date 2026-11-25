# srsr_nn: learned robustness mask for SRSR, trained through the exact merge

Goal: a small network that replaces SRSR's analytic rejection + dilate mask,
rejects ghosts the analytic mask keeps, and still runs once per frame on an
iPhone 15 at the speed and memory of today's pipeline.

Approach (after the IPOL Wronski et al. implementation's author's advice, and
DBSR, Bhat et al. 2021):
- Train end to end on the FINAL image, not on hand-made mask labels.
- The fusion is SRSR's own merge, made differentiable exactly. The merge samples
  each frame's mask once per output pixel and multiplies all of that frame's taps
  by it, so
  `out = (A_ref + Σ_f R_f·A_f) / (B_ref + Σ_f R_f·B_f)`
  with A, B computed once per sample (`srsr_merge.py`, a line-by-line port of
  `srsr_merge` in core/SRSRKernels.metal).
- Parity against the kernel's own output: 99.99% bit-identical after the same
  half rounding; the rest differ by one half ULP (`parity.py`). This holds at
  1x and 2x.
- Flow, covariances and the analytic mask come from the real SRSR pipeline
  (`srsr_dump`: the app's .metal source run on the CPU), so the network learns
  the real aligner's errors.

## Pipeline

| step | file | what |
|---|---|---|
| scenes | `build/dng_dump <dng> data/scenes/<name>.f32 > data/scenes/<name>.meta` | an iPhone DNG through the app's loader |
| data | `gen.py` | synthetic bursts (tools/robust_nn/rnn/burst.py: exact motion, occlusion, moving objects; device CFA, WB, noise profile x ISO), run through SRSR, stored with the noise-free target |
| merge | `srsr_merge.py` | exact differentiable merge (full image or windows) |
| model | `model.py` | DBSR-style: shared encoder on variance-stabilised quads, bilinear warp by SRSR's flow, decoder -> correction of the analytic mask's logit (zero init = analytic mask). Sizes `big` (299k params), `small` (15k), `tiny` |
| train | `train.py` | noise-normalised Charbonnier + gradient loss, ghost pixels up-weighted (`data.ghost_map`). Refuses to train without `--go` |
| eval | `train.py --eval` | error on ghost / clean pixels and keep weights: net vs analytic vs all kept vs reference only |

Build the C++ tools (MinGW g++ + OpenMP, LibRaw built under vendor/LibRaw or `LIBRAW=`):

    sh build.sh

Data, then training (the user starts training):

    python gen.py --out data/train --n 2000 --seed0 0
    python gen.py --out data/val   --n 60   --seed0 9000000
    python train.py --data data/train --val data/val --model big --out runs/big1 --go
    python train.py --val data/val --init runs/big1/model.pt --model big --eval

Plan: train `big` to prove a learned mask beats the analytic one on ghosts
without losing merge weight on clean pixels. Then distil it into `small`/`tiny`,
port that to a Metal kernel in the SRSR pipeline (one pass per comparison
frame, fp16), and quantise if it holds.

## Notes

- Sample size: 768 raw px crops (the SRSR pyramid needs >= 8 px at 1/32
  resolution), ~20 MB per .npz.
- `ghost_map`: per output pixel, how much keeping the worst comparison frame
  makes the merge worse than the reference alone, in noise std. It uses only
  exact merge terms and the target, and no mask.
- Baseline on 6 samples: the analytic mask keeps 51% weight on ghost pixels.
  Its error there (1.75 sigma) is worse than the reference alone (1.64). On
  clean pixels it is near "all kept" (0.468 vs 0.480 sigma).
