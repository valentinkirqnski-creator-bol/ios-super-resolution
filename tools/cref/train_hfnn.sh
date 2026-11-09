#!/bin/sh
# HF-artifact NN training (reuses the cref v2 crops; no data generation), then export to
# core/hfnn_weights.h. Run ONLY on the user's go.
set -e
SP=/c/Users/valen/AppData/Local/Temp/claude/b742f57fdf29e2e2/75d997e5-7f0e-41b3-bcd6-cbef711f0996/scratchpad
cd $SP/cref
python -B train_hfnn.py --train "$(cygpath -w data/train)" --val "$(cygpath -w data/val)" \
    --out "$(cygpath -w runs/hfnn1)" --minutes ${MINUTES:-30}
python -B export_hfnn.py "$(cygpath -w runs/hfnn1/best.pt)" "$(cygpath -w $SP/wt_8a28f74/core/hfnn_weights.h)"
