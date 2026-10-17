import time
import torch

lib = torch.mps.compile_shader(open("kernels.metal").read())

def timed(fn, trials=50):
    """Median seconds per call, waiting for the GPU before each clock reading."""
    fn(); torch.mps.synchronize()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter(); fn(); torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[trials // 2]

def timed_queued(fn, reps=200):
    """Seconds per call when `reps` calls are queued back to back and the GPU
    is waited on once: throughput, as a training loop sees it."""
    fn(); torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / reps