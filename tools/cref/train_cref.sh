#!/bin/sh
# Consensus refiner training: 60 min CPU budget, then export the best checkpoint to core/cref_weights.h.
# Run ONLY on the user's go.
set -e
SP=/c/Users/valen/AppData/Local/Temp/claude/b742f57fdf29e2e2/75d997e5-7f0e-41b3-bcd6-cbef711f0996/scratchpad
cd $SP/cref
python -B train_cref.py --train "$(cygpath -w data/train)" --val "$(cygpath -w data/val)" \
    --out "$(cygpath -w runs/cref2)" --minutes ${MINUTES:-60} --tau ${TAU:-2.0}
python -B export_weights.py "$(cygpath -w runs/cref2/best.pt)" "$(cygpath -w $SP/wt_8a28f74/core/cref_weights.h)"
