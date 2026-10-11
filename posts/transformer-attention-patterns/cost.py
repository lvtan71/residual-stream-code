import time
import torch
import torch.nn.functional as F

# Attention's cost against sequence length: one layer's causal attention at
# Qwen2.5-1.5B's shapes (12 query heads sharing 2 key-value heads, 128
# dimensions each, 16-bit), timed for n = 512 to 8,192 tokens, written out
# (scores as an n x n matrix per head, then softmax, then the weighted sum).
# The score matrices' size is printed too. Time with nothing else on the GPU.
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(DEVICE, lambda: None)
H, KV, D = 12, 2, 128

def attend(q, k, v):
    k, v = k.repeat_interleave(H // KV, 0), v.repeat_interleave(H // KV, 0)
    n = q.shape[1]
    scores = q @ k.transpose(-1, -2) / D ** 0.5                  # [heads, n, n]
    scores = scores.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return F.softmax(scores, dim=-1) @ v

def timed(fn, trials=10):
    fn(); sync()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter(); fn(); sync()
        times.append(time.perf_counter() - t0)
    return sorted(times)[trials // 2]

torch.manual_seed(0)
print("tokens    time (ms)   x previous   score matrices (MiB)")
prev = None
for n in (512, 1024, 2048, 4096, 8192):
    q = torch.randn(H, n, D, device=DEVICE, dtype=torch.float16)
    k = torch.randn(KV, n, D, device=DEVICE, dtype=torch.float16)
    v = torch.randn(KV, n, D, device=DEVICE, dtype=torch.float16)
    t = timed(lambda: attend(q, k, v))
    ratio = f"{t / prev:8.1f}" if prev else "        "
    print(f"{n:6d}   {t * 1e3:10.2f}   {ratio}     {H * n * n * 2 / 2**20:12.0f}")
    prev = t
