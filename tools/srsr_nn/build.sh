#!/bin/sh
# Builds the PC tools of tools/srsr_nn (MinGW g++ with OpenMP; Windows or any POSIX box):
#   build/srsr_dump  one synthetic burst through SRSR (core/SRSRKernels.metal on the CPU) -> merge dump
#   build/dng_dump   one DNG through the app's loader -> scene file for gen.py
# LIBRAW: a built LibRaw checkout (lib/libraw.a); default vendor/LibRaw (scripts/fetch_libraw.sh).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
C=$ROOT/core
L=${LIBRAW:-$ROOT/vendor/LibRaw}
O=$HERE/build/obj
mkdir -p "$O"
F="-std=c++17 -O2 -w -fopenmp -DHAVE_LIBRAW=1 -DHHSR_WITH_SRSR -I$HERE/cpp/pc -I$L -I$L/libraw -I$C"
CORE="frame_gate align burst_consensus cref d3d11_merge dng_writer fb_confidence feat_metric finish_hdr flow_consensus
global_homography grey_pyramid hfnn fusion_nn isa_prealign kernels ljpeg mann merge pipeline pipeline_paths preset_lut raw_io
ref_demosaic render_isp robustness snr_tuning structure_map tuning_json srsr_pipeline srsr_cpu"
for f in $CORE; do g++ $F -c "$C/$f.cpp" -o "$O/$f.o" & done
g++ $F -c "$HERE/cpp/pc/platform_stubs.cpp" -o "$O/platform_stubs.o" &
g++ $F -c "$HERE/cpp/pc/prof_stubs.cpp" -o "$O/prof_stubs.o" &
g++ $F -c "$HERE/cpp/srsr_dump.cpp" -o "$HERE/build/srsr_dump.o" &
g++ $F -c "$ROOT/tools/robust_nn/cpp/dng_dump.cpp" -o "$HERE/build/dng_dump.o" &
wait
LIBS="$L/lib/libraw.a -static"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) LIBS="$LIBS -ld3d11 -ld3dcompiler -ldxgi -lws2_32" ;; esac
g++ -O2 -fopenmp -o "$HERE/build/srsr_dump" "$O"/*.o "$HERE/build/srsr_dump.o" $LIBS
g++ -O2 -fopenmp -o "$HERE/build/dng_dump" "$O"/*.o "$HERE/build/dng_dump.o" $LIBS
echo "built $HERE/build/srsr_dump $HERE/build/dng_dump"
