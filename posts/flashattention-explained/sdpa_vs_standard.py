import resource
import subprocess
import sys
import time
import torch
import torch.nn.functional as F

# Standard attention (score matrix, softmax, weighted sum: three operations)
# against PyTorch's fused scaled_dot_product_attention, which on the CPU
# runs a FlashAttention-style tiled kernel. One head of size 64, 32-bit
# floats. Each measurement runs in a fresh process so its peak memory can
# be read on its own. Run on an otherwise idle machine: under memory
# pressure the standard version's big matrices get swapped, and its times
# become meaningless.
def run(kind, n, d=64):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, n, d) for _ in range(3))
    def standard():
        s = q @ k.transpose(-1, -2) / d ** 0.5
        return torch.softmax(s, -1) @ v
    def fused():
        return F.scaled_dot_product_attention(q, k, v)
    f = standard if kind == "standard" else fused
    f()
    t0 = time.perf_counter()
    for _ in range(3):
        out = f()
    t = (time.perf_counter() - t0) / 3
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) / 2**20   # bytes on macOS, KiB on Linux -> MiB
    print(f"{t * 1e3:.1f} {peak:.0f}")

if len(sys.argv) > 1:
    torch.set_num_threads(4)
    run(sys.argv[1], int(sys.argv[2]))
else:
    base = None
    for n in (2048, 8192, 16384):
        row = {}
        for kind in ("standard", "fused"):
            out = subprocess.run([sys.executable, __file__, kind, str(n)], capture_output=True, text=True).stdout.split()
            row[kind] = (float(out[0]), float(out[1]))
        (ts, ms), (tf, mf) = row["standard"], row["fused"]
        print(f"n = {n:>6,}: standard {ts:7.1f} ms, peak {ms:6,.0f} MiB;  fused {tf:7.1f} ms, peak {mf:6,.0f} MiB;  "
              f"the n x n score matrix is {n * n * 4 / 2**20:5,.0f} MiB")
    torch.manual_seed(0)
    a = torch.randn(1, 1, 4096, 64); b = torch.randn(1, 1, 4096, 64); c = torch.randn(1, 1, 4096, 64)
    ref = torch.softmax(a @ b.transpose(-1, -2) / 8, -1) @ c
    print(f"largest difference between the two at n = 4,096: {(ref - F.scaled_dot_product_attention(a, b, c)).abs().max():.1e}")