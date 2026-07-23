"""Compare py_*.npy dumps to cpp_*.bin dumps (same core name).

Usage:
  set HHSR_DEBUG_DIR=path\\to\\folder_with_both
  python tools/compare_pipelines.py

Or with separate folders:
  python tools/compare_pipelines.py --py path\\to\\python --cpp path\\to\\cpp
"""
import argparse
import glob
import os
import sys

import numpy as np


def load_bin(filepath):
    return np.fromfile(filepath, dtype=np.float32)


def compare_dirs(py_dir, cpp_dir, tol=1e-4):
    py_files = sorted(glob.glob(os.path.join(py_dir, "py_*.npy")))
    if not py_files:
        print(f"No py_*.npy in {py_dir}")
        return

    print("=" * 72)
    print(f"{'name':<28} {'py':>10} {'cpp':>10} {'max|err|':>12} {'match':>8}")
    print("=" * 72)

    for py_file in py_files:
        basename = os.path.basename(py_file)
        core_name = basename[3:-4]  # strip py_ and .npy
        cpp_file = os.path.join(cpp_dir, f"cpp_{core_name}.bin")
        if not os.path.exists(cpp_file):
            print(f"{core_name:<28} {'ok':>10} {'MISSING':>10} {'-':>12} {'-':>8}")
            continue

        py_arr = np.load(py_file).astype(np.float32).ravel()
        cpp_arr = load_bin(cpp_file)
        n = min(py_arr.size, cpp_arr.size)
        if n == 0:
            print(f"{core_name:<28} {py_arr.size:>10} {cpp_arr.size:>10} {'empty':>12} {'-':>8}")
            continue
        diff = np.abs(py_arr[:n] - cpp_arr[:n])
        max_err = float(np.max(diff))
        match = "YES" if max_err < tol else "NO"
        size_note = "" if py_arr.size == cpp_arr.size else f" (trunc {n})"
        print(f"{core_name:<28} {py_arr.size:>10} {cpp_arr.size:>10} {max_err:>12.6g} {match:>8}{size_note}")

    print("=" * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--py", default=None, help="Folder with py_*.npy")
    p.add_argument("--cpp", default=None, help="Folder with cpp_*.bin")
    p.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args()

    if args.py or args.cpp:
        py_dir = args.py or args.cpp
        cpp_dir = args.cpp or args.py
    else:
        dump_dir = os.environ.get("HHSR_DEBUG_DIR", "debug_dumps")
        py_dir = cpp_dir = dump_dir

    if not os.path.isdir(py_dir):
        print(f"Python dump dir not found: {py_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(cpp_dir):
        print(f"C++ dump dir not found: {cpp_dir}", file=sys.stderr)
        sys.exit(1)

    compare_dirs(py_dir, cpp_dir, args.tol)


if __name__ == "__main__":
    main()
