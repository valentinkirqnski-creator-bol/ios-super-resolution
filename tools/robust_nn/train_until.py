"""Train v13 in rounds until held-out accuracy vs the ground-truth mask reaches the goal.

Accuracy = on pixels where the ground truth is clear (target < 0.2 misaligned / > 0.8 aligned), the fraction
where the mask (exact 0/1 stretch) makes the same call (R < 0.5 / R > 0.5). Each round continues from the
previous best; if two rounds in a row gain < 0.3 points the hidden layer is widened 16 -> 32 (same function at
the moment of widening). Log: runs/v13/rounds.log
"""
import os
import re
import subprocess
import sys

GOAL, MAX_ROUNDS, STEPS = 0.95, 12, 150
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = "data/rp13_train,data/rp13_train_hdrp"
VAL = "data/rp13_val"


def acc_lines(path):
    return [float(x) for x in re.findall(r"accuracy vs GT mask ([0-9.]+)", open(path).read())]


def main():
    os.makedirs(os.path.join(HERE, "runs", "v13"), exist_ok=True)
    log = open(os.path.join(HERE, "runs", "v13", "rounds.log"), "a")
    env = dict(os.environ, RNN_GEO="1", RNN_CORRUPT_P="0.8")
    prev, hid, best, stall = "runs/v12/model.pt", 16, 0.0, 0
    for r in range(1, MAX_ROUNDS + 1):
        out = f"runs/v13/r{r:02d}"
        cmd = [sys.executable, "real_protocol.py", "train", "--init", prev, "--train", TRAIN, "--val", VAL,
               "--out", out, "--steps", str(STEPS), "--eval_every", str(STEPS), "--lr", "1e-3", "--workers", "3",
               "--hid", str(hid)]
        with open(os.path.join(HERE, out + ".log"), "w") as fh:
            os.makedirs(os.path.join(HERE, out), exist_ok=True)
            subprocess.run(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
        accs = acc_lines(os.path.join(HERE, out + ".log"))
        start, end = accs[0], max(accs)
        msg = f"ROUND {r} hid={hid}: accuracy start {start:.4f} -> best {end:.4f}"
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()
        if end >= GOAL:
            print(f"GOAL REACHED: {end:.4f} (round {r}, {out}/model.pt)", flush=True)
            log.write(f"GOAL REACHED {end:.4f} {out}\n")
            return
        stall = stall + 1 if end - best < 0.003 else 0
        best = max(best, end)
        prev = f"{out}/model.pt"
        if stall >= 2 and hid < 32:
            hid, stall = 32, 0
            print("WIDENING hidden layer to 32 for the next round", flush=True)
            log.write("WIDEN 32\n")
    print(f"STOPPED after {MAX_ROUNDS} rounds, best accuracy {best:.4f}", flush=True)
    log.write(f"STOPPED best {best:.4f}\n")


if __name__ == "__main__":
    main()
