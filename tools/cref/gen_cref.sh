#!/bin/sh
# Consensus-refiner data (~25-30 min on 6 cores): camera-scale 1024x1024 8-frame bursts rendered from
# real DNG scenes (rnn/burst.py via real_protocol.py gen, RNN_LEAN=1), small content slips (RNN_ROT_SMALL /
# RNN_SMALL_DISP), half with burst9's measured hand motion. Then the app's aligner + injected tile slips +
# Alg.6 and GCam masks + consensus features + ghost-size labels (cref_dataset.exe), as 80 px crops.
# Training scenes exclude APC_1211 (validation) and APC_1294 (burst12, kept as a real test).
set -e
SP=/c/Users/valen/AppData/Local/Temp/claude/b742f57fdf29e2e2/75d997e5-7f0e-41b3-bcd6-cbef711f0996/scratchpad
G=$SP/cref/gen; D=$SP/cref/data
mkdir -p $D/train $D/val
export PYTHONDONTWRITEBYTECODE=1
export RNN_REAL="C:/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn/data/real"
export RNN_NFRAMES=8 RNN_TEX_SCALE=0.5 RNN_LEAN=1 RNN_SIZE=1024 RNN_ROT_SMALL=20,60 RNN_SMALL_DISP=0.1,3.0,0.5,0.7
SC="APC_1186 APC_1258 APC_1278 APC_1340 APC_1342 HDRP_N000"
t0=$(date +%s)
cd $G
python -B real_protocol.py gen --scenes $SC --out "$(cygpath -w $D/rpA)" --n ${NA:-64} --seed0 200000 --workers 6
RNN_MOTION="$(cygpath -w $SP/wt_8a28f74/tools/robust_nn/data/motion_burst9.txt)" \
  python -B real_protocol.py gen --scenes $SC --out "$(cygpath -w $D/rpM)" --n ${NM:-48} --seed0 300000 --workers 6
python -B real_protocol.py gen --val --out "$(cygpath -w $D/rpV)" --n ${NV:-10} --seed0 900000 --workers 6
echo "RENDER_OK $(( $(date +%s) - t0 ))s"
cd $SP/cref
one() {   # $1 npz, $2 out dir, $3 seed
    b=$(basename $1 .npz)
    python -B npz2bin.py "$(cygpath -w $1)" "$(cygpath -w $2/$b.bin)"
    OMP_NUM_THREADS=4 ./cref_dataset.exe $2/$b.bin $2/$b.crops $3 12 80 2>&1 | grep -E "^burst|^wrote" | tr '\n' ' '; echo
    rm -f $2/$b.bin
}
i=0
for f in $D/rpA/*.npz $D/rpM/*.npz; do
    i=$((i + 1)); one $f $D/train $i &
    [ $((i % 3)) -eq 0 ] && wait
done
wait
for f in $D/rpV/*.npz; do i=$((i + 1)); one $f $D/val $i & [ $((i % 3)) -eq 0 ] && wait; done
wait
rm -rf $D/rpA $D/rpM          # train renders are only needed for the crops (val renders kept for full-frame eval)
echo "CROPS_OK $(( $(date +%s) - t0 ))s"
du -sh $D/train $D/val
