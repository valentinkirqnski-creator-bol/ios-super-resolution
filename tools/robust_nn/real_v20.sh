#!/bin/sh
# Real-burst check of one robust-NN checkpoint through the app's CPU stages (burst_compare), with the kernel fix
# (flow features read the residual flow after the global homography).
#   sh real_v20.sh runs/v20/model_s12000.pt v20s12 [logit_shift] [crop]
# crop: "W H" centre crop (default 2048 1536), or "full" for the whole 4032x3024 frame (writes merges too).
set -e
cd "$(dirname "$0")"
MODEL=$1; TAG=$2; SHIFT=${3:-0}; CROP=${4:-"2048 1536"}
ROOT="$(cd ../.. && pwd)"
B=build/pipe2/$TAG
mkdir -p $B
python - "$MODEL" "$SHIFT" "$B/w.h" <<'EOF'
import sys, torch
sys.path.insert(0, ".")
from real_protocol import load_net
from export_weights import write_header
net = load_net(sys.argv[1])
with torch.no_grad():
    net.c3.bias += float(sys.argv[2])          # the logit shift, folded into the last bias (weights-only)
write_header(net, sys.argv[3], f"{sys.argv[1]} shift {sys.argv[2]}")
EOF
F="-O2 -std=gnu++17 -DHAVE_LIBRAW -I$ROOT/core -I$ROOT/tools/rob_refine -Icpp -I$ROOT/vendor/LibRaw -pthread -fopenmp"
H="$(pwd)/$B/w.h"
CORE="$(for o in align grey_pyramid host_stubs kernels merge raw_io robustness snr_tuning global_homography; do printf 'build/pipe2/%s.o ' $o; done)"
g++ $F "-DRNN_WEIGHTS_HEADER=\"$H\"" -c cpp/robust_nn.cpp -o $B/robust_nn.o
g++ $F "-DRNN_WEIGHTS_HEADER=\"$H\"" -c cpp/burst_compare.cpp -o $B/burst_compare.o
g++ $F -o $B/burst_compare.exe $B/burst_compare.o $B/robust_nn.o $CORE $ROOT/vendor/LibRaw/lib/libraw.a -lws2_32 -static
for BURST in burst11 burst9; do
    O=data/real_cmp/${TAG}_$BURST; mkdir -p $O
    if [ "$CROP" = "full" ]; then XY="0 0 4032 3024"; M=nn; else XY="-1 -1 $CROP"; M=none; fi
    unset BC_HOMOG; [ $BURST = burst11 ] && export BC_HOMOG=1   # an empty value would still switch it on
    BC_RESID_FEAT=1 BC_GAIN_NORM=1 BC_MERGES=$M ./$B/burst_compare.exe $O $XY 1.0 0.2 0.9 \
        $(ls /c/Users/valen/Downloads/$BURST/*.dng) 2> $O/log.txt
    echo "$TAG $BURST: $(grep 'mean R' $O/log.txt | sed 's/.*nn //' | tr '\n' ' ')"
done
