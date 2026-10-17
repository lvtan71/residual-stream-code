import time
import torch

def gelu_scale_shift(x, g, b):
    """Four elementwise steps. Eager PyTorch runs each as its own pass over memory."""
    return torch.nn.functional.gelu(x) * g + b + x

def scale_shift(x, g, b):
    """The same without GELU: three steps, almost no arithmetic."""
    return x * g + b + x

x = torch.randn(4096, 4096)
g, b = torch.randn(4096), torch.randn(4096)

def timed(fn, trials=20):
    fn(); times = []
    for _ in range(trials):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    return sorted(times)[trials // 2]

for name, fn in [("GELU, scale, shift, add", gelu_scale_shift), ("scale, shift, add", scale_shift)]:
    t0 = time.perf_counter()
    fast = torch.compile(fn)
    fast(x, g, b)
    compile_s = time.perf_counter() - t0
    eager_s, fused_s = timed(lambda: fn(x, g, b)), timed(lambda: fast(x, g, b))
    same = torch.allclose(fast(x, g, b), fn(x, g, b), atol=1e-5)
    print(f"{name:24s} eager {eager_s * 1e3:5.1f} ms   compiled {fused_s * 1e3:4.1f} ms "
          f"({eager_s / fused_s:.1f}x)   first call {compile_s:.1f} s   same result: {same}")
t = timed(lambda: x.clone())
print(f"{'one read + one write':24s} {t * 1e3:5.1f} ms  (x.clone(), the floor for any one-pass kernel)")