import math
import numpy as np

# How many GPUs does the LLM stage of 1,000 simultaneous voice calls need?
# An idealized simulation of continuous batching ("Serving Metrics and Prefill–Decode Disaggregation"), one iteration
# at a time, for an 8B model in BF16 on H100s (80 GB).
#   Calls: each sends the LLM a request every ~20 s on average (the caller
#   talks for a few seconds; the agent speaks its ~45-word answer for ~17 s);
#   arrivals are random (Poisson), 50 per second in all.
#   A request: a 2,000-token conversation so far plus 150 new tokens, and
#   60 output tokens. Each call is pinned to one GPU, so its cache is there.
#   An iteration decodes one token for every active request and prefills the
#   requests admitted this iteration. Its time is the larger of reading the
#   weights and every KV cache it touches at 3.35 TB/s, and the arithmetic at
#   40% of 989 TFLOPS.
# Where the 2,000 tokens of each call's history live between turns decides
# everything (2,000 tokens x 128 KiB = 262 MB per call, 262 GB for 1,000):
#   "HBM"     : kept in GPU memory, so only the 150 new tokens are prefilled,
#               but every call's cache must fit beside the weights;
#   "re-prefill": nothing kept, all 2,150 tokens prefilled every turn;
#   "CPU"     : kept in host memory and copied back over PCIe (~25 GB/s
#               effective) when the call's next request arrives.
N, W, BW, FLOPS, HBM = 8e9, 16e9, 3.35e12, 0.4 * 989e12, 80e9
KV, CONTEXT, NEW, OUT, MAX_BATCH, PCIE = 131072, 2000, 150, 60, 200, 25e9
CALLS, TURN_S = 1000, 20.0

def simulate(gpus, mode, seconds=120, seed=0):
    calls_per_gpu = math.ceil(CALLS / gpus)
    if mode == "HBM" and W + calls_per_gpu * CONTEXT * KV + MAX_BATCH * (NEW + OUT) * KV > 0.9 * HBM:
        return None                                          # the caches don't fit
    prefill = NEW if mode != "re-prefill" else CONTEXT + NEW
    reload = CONTEXT * KV / PCIE if mode == "CPU" else 0.0  # copy the history back before prefill
    rng = np.random.default_rng(seed)
    t, arrivals = 0.0, []
    while t < seconds:
        t += rng.exponential(TURN_S / CALLS)
        arrivals.append((t + reload, int(rng.integers(CALLS))))  # (ready time, call)
    arrivals.sort(reverse=True)
    clock, waiting, active = [0.0] * gpus, [[] for _ in range(gpus)], [[] for _ in range(gpus)]
    ttft, itl = [], []
    while arrivals or any(waiting) or any(active):
        g = min(range(gpus), key=lambda i: clock[i])
        now = clock[g]
        while arrivals and arrivals[-1][0] <= now:
            ready, call = arrivals.pop()
            waiting[call % gpus].append(ready - reload)       # sticky: a call always goes to its GPU
        if not waiting[g] and not active[g]:
            nxt = [a for a, c in arrivals if c % gpus == g]
            clock[g] = min(nxt) if nxt else float("inf")
            if all(c == float("inf") for c in clock):
                break
            continue
        admit = waiting[g][: max(0, MAX_BATCH - len(active[g]))]
        waiting[g] = waiting[g][len(admit):]
        reads = W + (len(active[g]) + len(admit)) * CONTEXT * KV        # weights + every cache touched
        dt = max(reads / BW, 2 * N * (len(active[g]) + prefill * len(admit)) / FLOPS)
        clock[g] = now + dt
        if active[g]:
            itl.append(dt)
        active[g] = [r - 1 for r in active[g] if r > 1] + [OUT - 1] * len(admit)
        ttft += [clock[g] - a for a in admit]
    warm = len(ttft) // 10
    return np.array(ttft[warm:]), np.array(itl)

print(f"{CALLS} calls, a request every {TURN_S:.0f} s per call; 2,000 + 150 prompt tokens, {OUT} out")
print(f"{'history kept in':>15s} {'GPUs':>4s} {'TTFT p50':>9s} {'TTFT p99':>9s} {'ITL p50':>8s} {'ITL p99':>8s}")
for mode in ("HBM", "re-prefill", "CPU"):
    for gpus in (1, 2, 4, 6, 8, 12):
        r = simulate(gpus, mode)
        if r is None:
            print(f"{mode:>15s} {gpus:4d}   {-(-1000 // gpus)} calls' caches don't fit in 80 GB")
            continue
        ttft, itl = r
        if np.median(ttft) > 10:
            print(f"{mode:>15s} {gpus:4d}   overloaded: requests queue without bound")
            continue
        print(f"{mode:>15s} {gpus:4d} {np.median(ttft) * 1e3:6.0f} ms {np.percentile(ttft, 99) * 1e3:6.0f} ms "
              f"{np.median(itl) * 1e3:5.1f} ms {np.percentile(itl, 99) * 1e3:5.1f} ms", flush=True)