import torch
from gpu_util import timed_queued as timed

# Operations from very little math per byte to a lot, measured on this GPU in
# float32, each queued 200 times and waited on once (throughput). Intensity = FLOPs / bytes that must move between memory and the chip.
rows = []
x = torch.randn(1 << 24, device="mps"); y = torch.randn_like(x)
t = timed(lambda: x + y)
rows.append(("vector add", len(x), 3 * 4 * len(x), t))
for n in (4096, 8192):
    M, v = torch.randn(n, n, device="mps"), torch.randn(n, device="mps")
    t = timed(lambda: M @ v)
    rows.append((f"matrix-vector, n={n}", 2 * n * n, 4 * (n * n + 2 * n), t))
for n in (64, 128, 256, 512, 1024, 2048, 4096):
    A, B = torch.randn(n, n, device="mps"), torch.randn(n, n, device="mps")
    t = timed(lambda: A @ B)
    rows.append((f"matmul, n={n}", 2 * n ** 3, 3 * 4 * n * n, t))
# The smallest matmul again, but 1,024 of them batched into one call.
A, B = torch.randn(1024, 64, 64, device="mps"), torch.randn(1024, 64, 64, device="mps")
t = timed(lambda: torch.bmm(A, B))
rows.append(("1,024 matmuls n=64, bmm", 1024 * 2 * 64 ** 3, 1024 * 3 * 4 * 64 * 64, t))
for name, flops, bytes_, t in rows:
    print(f"{name:22s} intensity {flops / bytes_:8.2f} FLOP/byte   {flops / t / 1e9:7.1f} GFLOP/s   "
          f"{bytes_ / t / 1e9:6.1f} GB/s")