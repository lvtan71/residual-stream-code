import time
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
gpu_sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(DEVICE, lambda: None)   # the post ran on "mps"

def timed(fn, device, trials=10):
    """Median seconds per call. GPU work is queued, not finished, when a call returns,
    so wait for the device before reading the clock."""
    sync = gpu_sync if device == DEVICE else (lambda: None)
    fn(); sync()                                        # warm up
    times = []
    for _ in range(trials):
        t0 = time.perf_counter(); fn(); sync()
        times.append(time.perf_counter() - t0)
    return sorted(times)[trials // 2]

cases = [("cpu", torch.float32), (DEVICE, torch.float32), (DEVICE, torch.float16), (DEVICE, torch.bfloat16)]
for n in (256, 1024, 4096):
    row = [f"n = {n:4d}:"]
    for device, dt in cases:
        a = torch.randn(n, n, device=device, dtype=dt)
        b = torch.randn(n, n, device=device, dtype=dt)
        s = timed(lambda: a @ b, device)
        row.append(f"{device} {str(dt).split('.')[1]:8s} {2 * n ** 3 / s / 1e12:5.2f}")
    print("  ".join(row) + "  TFLOP/s", flush=True)

# The same GPU matmul, timed without waiting for it: the clock only sees the launch.
a = torch.randn(4096, 4096, device=DEVICE)
a @ a; gpu_sync()
t0 = time.perf_counter(); a @ a; t_launch = time.perf_counter() - t0
gpu_sync()
print(f"4096 matmul on mps, no synchronize: {t_launch * 1e3:.2f} ms 'measured', "
      f"{timed(lambda: a @ a, DEVICE) * 1e3:.2f} ms real")

# Synchronizing after every call also measures the round trip to the GPU.
# Queue many calls, synchronize once, and the GPU can overlap them.
a = torch.randn(256, 256, device=DEVICE)
a @ a; gpu_sync()
t0 = time.perf_counter()
for _ in range(200):
    a @ a
gpu_sync()
queued = (time.perf_counter() - t0) / 200
print(f"256 matmul on mps: {timed(lambda: a @ a, DEVICE) * 1e6:.0f} us per call waiting each time, "
      f"{queued * 1e6:.0f} us per call when 200 are queued ({2 * 256 ** 3 / queued / 1e12:.2f} TFLOP/s)")