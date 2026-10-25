#!/bin/sh
# Real-burst check of the given models (args: checkpoint paths) on the chosen regions of the three unseen bursts.
# Caller runs it at below-normal priority. Outputs: ~/Downloads/mfsr_conf_outputs/<burst>_x.._y.._<w>/
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd "$(dirname "$0")"
D=/c/Users/valen/Downloads
run() { python -W ignore real_eval.py "$@" --threads 4 >> runs/real_eval.log 2>&1; }
for H in "" "--homog"; do
  run "$@" --burst $D/burst9 --ref APC_1186 --box 2032 1536 256 256 $H
  run "$@" --burst $D/burst9 --ref APC_1186 --box 700 1100 384 384 $H
  run "$@" --burst $D/burst11 --ref APC_1278 --box 1500 1300 384 384 $H
  run "$@" --burst $D/burst11 --ref APC_1278 --box 400 1150 384 384 $H
  run "$@" --burst $D/burst11 --ref APC_1278 --box 2900 600 384 384 $H
  run "$@" --burst $D/burst12 --ref APC_1294 --box 700 1100 384 384 $H
  run "$@" --burst $D/burst12 --ref APC_1294 --box 1800 1550 384 384 $H
  run "$@" --burst $D/burst12 --ref APC_1294 --box 2500 1900 384 384 $H
done
echo REAL_DONE >> runs/real_eval.log
