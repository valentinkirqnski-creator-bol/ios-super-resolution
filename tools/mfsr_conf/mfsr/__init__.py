"""mfsr_conf: a neural confidence estimator for the robustness mask of the app's handheld MFSR merge.

Modules
  config      YAML configs with defaults (configs/*.yaml)
  scenes      scene sources (real DNG scenes per RGGB quad, procedural text / charts), split by source image
  formation   image formation: layered world -> optics -> pixel aperture -> CFA -> noise -> clip -> loader domain,
              exact ground-truth correspondence + visibility, HR target on the merge's output grid
  scenarios   evaluation categories / training mix -> world + alignment + corruption settings
  corrupt     deliberate alignment errors (tile offsets, smooth fields, rotation, blocks, edge slips, ...)
  harness     the app's own stages (aligner, analytic mask, kernel covariances) via build/conf_harness.exe
  merge       exact PyTorch replica of the app's merge (merge.cpp) + recombination for any R map + cost
  labels      alignment-confidence target (known correspondence error) and merge-confidence target
              (reconstruction-optimal R from the HR target), include/exclude utility
  inputs      per-frame network inputs (noise-normalised, guide + raw resolution)
  model       ConfNet: shared per-frame encoder, cross-frame fusion, two heads, raw-resolution output
  data        dataset of generated samples, patch extraction
  metrics     reconstruction-based evaluation
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(TOOL))
ROBUST_NN = os.path.join(ROOT, "tools", "robust_nn")
if ROBUST_NN not in sys.path:
    sys.path.append(ROBUST_NN)          # reuse rnn.* (world model, features) from the existing tool (appended: its
                                        # train.py / evaluate.py must not shadow this tool's)
