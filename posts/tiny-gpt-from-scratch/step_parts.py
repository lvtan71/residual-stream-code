import time
import torch
import torch.nn.functional as F
from gpt import GPT, Config

# Where a training step's 150 ms go: the model's matrix multiplies alone,
# at the step's real shapes (8,192 rows), forward and backward; attention's
# fused call alone, with and without dropout; and AdamW's update.
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(dev, lambda: None)   # the post ran on "mps"
def timed(fn, n=20):
    for _ in range(3):
        fn()
    sync(); t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0) / n * 1e3

c = Config(V=148)
M, D, F_ = 32 * c.T, c.D, 512
shapes = [(D, 3 * D), (D, D), (D, F_), (D, F_), (F_, D)] * c.L + [(D, c.V)]   # per layer, then unembedding
xs = [torch.randn(M, k, device=dev, requires_grad=True) for k, _ in shapes]
ws = [torch.randn(k, n, device=dev, requires_grad=True) for k, n in shapes]
def matmuls():
    sum((x @ w).sum() for x, w in zip(xs, ws)).backward()
flops = sum(6 * M * k * n for k, n in shapes)                    # 2 forward + 4 backward per multiply-add
t = timed(matmuls)
print(f"the {len(shapes)} matrix multiplies alone, forward and backward: {t:.0f} ms, {flops / t / 1e6:.0f} GFLOP/s")

q, k, v = (torch.randn(32, c.N, c.T, D // c.N, device=dev, requires_grad=True) for _ in range(3))
for p in (0.0, 0.2):
    def attn():
        F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p).sum().backward()
    print(f"attention's fused call, {c.L} layers, dropout {p}: {c.L * timed(attn):.0f} ms")

m = GPT(c).to(dev)
opt = torch.optim.AdamW(m.parameters(), 1e-3)
for p in m.parameters():
    p.grad = torch.zeros_like(p)
print(f"AdamW's update of all parameters: {timed(opt.step):.1f} ms")