#!/bin/sh
# Second queue: runs after train_queue.sh finishes (QUEUE_DONE). Caller runs it at below-normal priority.
export PATH="/usr/bin:/mingw64/bin:$PATH"
cd "$(dirname "$0")"
until grep -q QUEUE_DONE runs/queue.log 2>/dev/null; do sleep 30; done
for n in m10_balanced; do
    [ -f runs/$n/DONE ] && continue
    mkdir -p runs/$n
    echo "start $n $(date)" >> runs/queue.log
    python -W ignore train.py configs/train_$n.yaml > runs/$n/stdout.log 2>&1 && touch runs/$n/DONE
    echo "end $n rc=$? $(date)" >> runs/queue.log
done
echo QUEUE2_DONE >> runs/queue.log
