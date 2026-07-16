import torch
import numpy as np

# Test PyTorch FFT
a = torch.tensor([1., 2., 3., 4.], dtype=torch.float32)
A = torch.fft.fft(a)
print("PyTorch forward FFT [1,2,3,4]:", A)

# Test PyTorch IFFT
B = torch.fft.ifft(A)
print("PyTorch inverse FFT:", B)

b = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float32)
B5 = torch.fft.fft(b)
print("PyTorch forward FFT [1,2,3,4,5]:", B5)
