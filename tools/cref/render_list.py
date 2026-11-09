"""Render a list of bursts one by one in this process (no multiprocessing Pool: on this Windows Python a dead
pool worker hangs the pool forever). usage: python render_list.py out_dir val(0/1) seed seed ...
Scenes: the 6 training scenes (ignored for val, which uses APC_1211)."""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen"))
import real_protocol as rp  # noqa: E402

SCENES = ["APC_1186", "APC_1258", "APC_1278", "APC_1340", "APC_1342", "HDRP_N000"]
out, val = sys.argv[1], sys.argv[2] == "1"
os.makedirs(out, exist_ok=True)
for s in (int(x) for x in sys.argv[3:]):
    t = time.time()
    try:
        rp._gen_one((s, out, val, SCENES))
        print(f"ok {s} {time.time() - t:.0f}s", flush=True)
    except Exception:
        print(f"FAIL {s}", flush=True)
        traceback.print_exc()
