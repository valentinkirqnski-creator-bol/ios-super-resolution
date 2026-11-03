#!/bin/sh
# burst9 (no homography, as the app runs by default) through the harness with v21 and v37: NN merge + masks each.
set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
F="-O2 -std=gnu++17 -DHAVE_LIBRAW -I$ROOT/core -I$ROOT/tools/rob_refine -Icpp -I$ROOT/vendor/LibRaw -pthread -fopenmp"
CORE="$(for o in align grey_pyramid host_stubs kernels merge raw_io robustness snr_tuning global_homography; do printf 'build/pipe2/%s.o ' $o; done)"
for TAG in v21 v37; do
    case $TAG in v21) MODEL=runs/v21/model.pt;; v37) MODEL=runs/v37/model_s8884.pt;; esac
    B=build/pipe2/${TAG}h; mkdir -p $B
    python - "$MODEL" "$B/w.h" <<'PY'
import sys
sys.path.insert(0, ".")
from real_protocol import load_net
from export_weights import write_header
write_header(load_net(sys.argv[1]), sys.argv[2], f"{sys.argv[1]}")
PY
    H="$(pwd)/$B/w.h"
    g++ $F "-DRNN_WEIGHTS_HEADER=\"$H\"" -c cpp/robust_nn.cpp -o $B/robust_nn.o
    g++ $F "-DRNN_WEIGHTS_HEADER=\"$H\"" -c cpp/burst_compare.cpp -o $B/burst_compare.o
    g++ $F -o $B/burst_compare.exe $B/burst_compare.o $B/robust_nn.o $CORE $ROOT/vendor/LibRaw/lib/libraw.a -lws2_32 -static
    O=data/real_cmp/${TAG}h_burst9; mkdir -p $O
    unset BC_HOMOG
    BC_RESID_FEAT=1 BC_GAIN_NORM=1 BC_MERGES=nn ./$B/burst_compare.exe $O 0 0 4032 3024 1.0 0.2 0.9 $(ls /c/Users/valen/Downloads/burst9/*.dng) 2> $O/log.txt
    echo "$TAG: $(grep 'mean R' $O/log.txt | sed 's/.*mean R //' | tr '\n' ' ')"
done
echo B9CMP_OK
