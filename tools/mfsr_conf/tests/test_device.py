"""Device code against PyTorch (needs g++ on PATH and generated data):
  - core/conf_nn_shared.h compiled as C++ (inputs, network, raw refinement, R) for three architecture variants
  - the app's ConfNet Metal kernels (exact source of core/HHSRKernels.metal through an MSL shim + the host band loop
    of metal_gpu.mm, NaN-poisoned band buffers) at band heights 32 and 7
"""
import glob
import os
import shutil
import subprocess
import sys
import unittest

TOOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Device(unittest.TestCase):
    def setUp(self):
        if not shutil.which("g++"):
            self.skipTest("g++ not on PATH")
        if not glob.glob(os.path.join(TOOL, "data", "d1", "val", "*.npz")):
            self.skipTest("no generated data")

    def run_script(self, script, *args):
        r = subprocess.run([sys.executable, "-W", "ignore", os.path.join(TOOL, script), *args], cwd=TOOL,
                           capture_output=True, text=True, timeout=1800)
        return r.stdout + r.stderr

    def test_shared_header_parity(self):
        out = self.run_script("parity_cnn.py")
        self.assertIn("PARITY OK", out, out[-2000:])

    def test_metal_kernels_emulated(self):
        out = self.run_script("kernel_test.py", "--samples", "1")
        self.assertIn("KERNELS OK", out, out[-2000:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
