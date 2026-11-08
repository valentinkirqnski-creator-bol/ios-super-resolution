# Consensus refiner (cref) — data + training

Small CNN (~2.2k weights, core/cref.h) that refines the finished Wronski/Alg.6 or GCam
robustness mask from each frame's noise-normalised fine structure compared with the
reference and with the leave-one-out consensus of the other aligned frames.
Output: R' = sigmoid(logit(R) + delta).

- `gen_cref.sh` / `finish_and_train.sh`: render camera-scale 8-frame bursts from the
  real scenes in tools/robust_nn/data/real. This uses tools/robust_nn's real_protocol.py,
  with the scene directory taken from env RNN_REAL. The scripts then convert the bursts
  (`npz2bin.py`) and crop them with `cref_dataset.cpp`.
  - `cref_dataset.cpp` runs the app aligner, injects tile slips, and builds the Alg.6 and
    GCam masks and the 6 features.
  - Its label is z = ghost size / one-frame noise.
  - Val scene: APC_1211. APC_1294 (burst12) is held out.
- `train_cref.py` + `train_cref.sh`: train within a time budget (default 60 min, CPU),
  then `export_weights.py` writes core/cref_weights.h.
- `cref_parity.cpp` / `.py` + `cref_kernels_emu.inc`: check that torch, the CPU
  implementation and the Metal kernels (run through a C++ shim) agree, using random
  weights.

The scripts carry absolute scratchpad paths from the machine they were developed on.
Edit SP / RNN_REAL before rerunning. Python's multiprocessing Pool on Windows (3.14)
can hang after a worker dies; render single bursts with `real_protocol._gen_one`
if that happens.
