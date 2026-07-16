import numpy as np
from scipy.ndimage._filters import _gaussian_kernel1d

# factor = 2, sigma = 1.0, radius = 4
kernel = _gaussian_kernel1d(sigma=1.0, order=0, radius=4)[::-1].copy()

print("Python Kernel:")
for v in kernel:
    print(f"{v:.8f}")
print("Sum:", sum(kernel))
