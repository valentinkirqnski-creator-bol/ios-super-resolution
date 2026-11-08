#!/bin/sh
# Finish the cref data (6 val bursts as independent single-burst renders, so one crashed worker can't
# hang the pool), build crops from the 73 already-rendered train bursts, then train for 60 min.
SP=/c/Users/valen/AppData/Local/Temp/claude/b742f57fdf29e2e2/75d997e5-7f0e-41b3-bcd6-cbef711f0996/scratchpad
G=$SP/cref/gen; D=$SP/cref/data
mkdir -p $D/train $D/val $D/rpV
export PYTHONDONTWRITEBYTECODE=1
export RNN_REAL="C:/Users/valen/Downloads/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/ios-super-resolution-1d39c2446f8978fdc5caedcd5b52a6d6bc746a85/tools/robust_nn/data/real"
export RNN_NFRAMES=8 RNN_TEX_SCALE=0.5 RNN_LEAN=1 RNN_SIZE=1024 RNN_ROT_SMALL=20,60 RNN_SMALL_DISP=0.1,3.0,0.5,0.7
t0=$(date +%s)
cd $G
for s in 900000 900001 900002 900003 900004 900005; do
    timeout 900 python -B real_protocol.py gen --val --out "$(cygpath -w $D/rpV)" --n 1 --seed0 $s --workers 1 > /dev/null 2>&1 &
done
wait
echo "VAL_RENDER $(ls $D/rpV/*.npz | wc -l) bursts $(( $(date +%s) - t0 ))s"
cd $SP/cref
one() {
    b=$(basename $1 .npz)
    python -B npz2bin.py "$(cygpath -w $1)" "$(cygpath -w $2/$b.bin)"
    OMP_NUM_THREADS=2 ./cref_dataset.exe $2/$b.bin $2/$b.crops $3 12 80 > /dev/null 2>&1
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
echo "CROPS_OK train $(ls $D/train/*.crops | wc -l) val $(ls $D/val/*.crops | wc -l) $(( $(date +%s) - t0 ))s"
du -sh $D/train $D/val
python -B crops.py "$(cygpath -w $D/train)" 2>&1 | tail -8
python -B crops.py "$(cygpath -w $D/val)" 2>&1 | tail -8
echo "TRAIN_START $(date)"
sh train_cref.sh 2>&1
echo "TRAIN_DONE $(date) rc=$?"
