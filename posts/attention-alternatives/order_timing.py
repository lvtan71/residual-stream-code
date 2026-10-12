"""Time the two multiplication orders of (non-causal) linear attention.

(Q K^T) V forms an n x n intermediate; Q (K^T V) forms a d x d one.
Counted FLOPs say they break even at n = d. This measures where they break
even on real hardware. Run it with nothing else using the machine.
"""

import time

import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device, lambda: None)
torch.manual_seed(0)
d = 128


def best_of(f, repeats=20):
    f()
    sync()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        f()
        sync()
        times.append(time.perf_counter() - start)
    return min(times)


print(f"device {device}, float32, d = {d}, best of 20 runs")
print(f"{'n':>7s} {'(QK^T)V':>10s} {'Q(K^T V)':>10s} {'measured':>9s} {'counted':>8s}")
for n in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
    Q, K, V = (torch.randn(n, d, device=device) for _ in range(3))
    quad = best_of(lambda: (Q @ K.T) @ V)
    lin = best_of(lambda: Q @ (K.T @ V))
    print(f"{n:>7,} {quad * 1e3:>8.3f}ms {lin * 1e3:>8.3f}ms {quad / lin:>8.1f}x {n / d:>7.2f}x")
