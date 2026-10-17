import torch
from gpu_util import lib, timed

n = 1 << 26                                             # 64M floats, 268 MB per array
a, b = torch.randn(n, device="mps"), torch.randn(n, device="mps")
out = torch.empty_like(a)
t = timed(lambda: lib.vector_add(out, a, b))
print(f"vector_add: read 2 arrays, write 1, in {t * 1e3:.1f} ms = {3 * n * 4 / t / 1e9:.0f} GB/s")
t = timed(lambda: out.copy_(a))
print(f"PyTorch copy: {2 * n * 4 / t / 1e9:.0f} GB/s")
# Odd strides are coprime with n, so every element is still read exactly once.
for stride in (1, 3, 9, 33, 1025):
    t = timed(lambda: lib.strided_copy(out, a, n, stride))
    print(f"strided_copy, stride {stride:4d}: {2 * n * 4 / t / 1e9:5.0f} GB/s")