#!/bin/sh
# Waits for dataset d1, then trains the main model and the ablations one after the other (caller runs this at
# below-normal priority). Each run logs to runs/<name>/stdout.log; the queue log is runs/queue.log.
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd "$(dirname "$0")"
mkdir -p runs
until grep -q GEN_DONE data/d1/gen.log 2>/dev/null; do sleep 30; done
echo "data ready $(date)" >> runs/queue.log
for n in m1 m4_refine m9_guideres m2_nofusion m5_norecon m6_recon4 m3_smallrf m7_crop64 m8_crop128; do
    [ -f runs/$n/best.pt ] && [ -f runs/$n/DONE ] && continue
    mkdir -p runs/$n
    echo "start $n $(date)" >> runs/queue.log
    python -W ignore train.py configs/train_$n.yaml > runs/$n/stdout.log 2>&1 && touch runs/$n/DONE
    echo "end $n rc=$? $(date)" >> runs/queue.log
done
echo QUEUE_DONE >> runs/queue.log
