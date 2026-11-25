#!/bin/sh
# Builds build/wronski_dump (MinGW g++ + OpenMP) from the app's core/.
# LIBRAW: a built LibRaw checkout (lib/libraw.a); default vendor/LibRaw.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
C=$ROOT/core
PC=$ROOT/tools/srsr_nn/cpp/pc            # PC stubs (platform, profiling, zlib)
L=${LIBRAW:-$ROOT/vendor/LibRaw}
O=$HERE/build/obj
mkdir -p "$O"
F="-std=c++17 -O2 -w -fopenmp -DHAVE_LIBRAW=1 -DHHSR_WITH_SRSR -I$PC -I$L -I$L/libraw -I$C"
CORE="frame_gate align burst_consensus cref d3d11_merge dng_writer fb_confidence feat_metric finish_hdr flow_consensus
global_homography grey_pyramid hfnn fusion_nn isa_prealign kernels ljpeg mann merge pipeline pipeline_paths preset_lut raw_io
ref_demosaic render_isp robustness snr_tuning structure_map tuning_json srsr_pipeline srsr_cpu"
for f in $CORE; do g++ $F -c "$C/$f.cpp" -o "$O/$f.o" & done
g++ $F -c "$PC/platform_stubs.cpp" -o "$O/platform_stubs.o" &
g++ $F -c "$PC/prof_stubs.cpp" -o "$O/prof_stubs.o" &
g++ $F -c "$HERE/cpp/wronski_dump.cpp" -o "$HERE/build/wronski_dump.o" &
wait
LIBS="$L/lib/libraw.a -static"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) LIBS="$LIBS -ld3d11 -ld3dcompiler -ldxgi -lws2_32" ;; esac
g++ -O2 -fopenmp -o "$HERE/build/wronski_dump" "$O"/*.o "$HERE/build/wronski_dump.o" $LIBS
echo "built $HERE/build/wronski_dump"
