import numpy as np

# Serving one 8B model (Llama-3-8B shapes: 32 layers, 8 KV heads of 128,
# BF16) on H100s, simulated step by step. A step processes some prompt
# tokens (prefill) and one new token for every running sequence (decode);
# its time is the roofline of "How a GPU Runs Deep Learning": the larger of reading the weights
# and the KV cache it touches, and doing its FLOPs at half of peak.
N, BW, FLOPS = 8.0e9, 3.35e12, 0.5 * 989e12
KV_TOKEN = 2 * 32 * 8 * 128 * 2                         # bytes per token of KV cache (128 KiB)

def step_time(prefill_tokens, decode_seqs, decode_ctx):
    """Seconds for one step: prefill_tokens prompt tokens, decode_seqs
    sequences generating one token each with decode_ctx total context."""
    flops = 2 * N * (prefill_tokens + decode_seqs)
    bytes_ = 2 * N + KV_TOKEN * decode_ctx
    return max(bytes_ / BW, flops / FLOPS)

rng = np.random.default_rng(0)
def workload(rate, n=2000):
    """Poisson arrivals; prompts log-normal around 2,000 tokens (200 to 16k),
    256 output tokens each."""
    t = np.cumsum(rng.exponential(1 / rate, n))
    p = np.clip(rng.lognormal(np.log(2000), 0.8, n), 200, 16000).astype(int)
    return [(float(t[i]), int(p[i]), 256) for i in range(n)]

def colocated(reqs, chunk=None, gpus=1):
    """Continuous batching on `gpus` independent GPUs (requests split round
    robin). chunk=None: a waiting prompt is prefilled whole, in the next
    step, stalling the running decodes for that step; chunk=c: at most c
    prompt tokens per step, mixed with the decodes (chunked prefill)."""
    out = []
    for g in range(gpus):
        q = list(reqs[g::gpus])
        t, waiting, running, i = 0.0, [], [], 0      # running: [ctx, left, arrival, first token, last token, max gap]
        while i < len(q) or waiting or running:
            while i < len(q) and q[i][0] <= t:
                waiting.append([q[i][0], q[i][1], q[i][1], q[i][2]]); i += 1   # arrival, prompt, left to prefill, out
            if not waiting and not running:
                t = q[i][0]; continue
            budget = chunk if chunk else (waiting[0][2] if waiting else 0)
            pre, done = 0, []
            for w in waiting:
                take = min(w[2], budget - pre)
                if take <= 0:
                    break
                w[2] -= take; pre += take
                if w[2] == 0:
                    done.append(w)
            ctx = sum(r[0] for r in running)
            t += step_time(pre, len(running), ctx)
            for r in running:
                r[0] += 1; r[1] -= 1
                r[5] = max(r[5], t - r[4]); r[4] = t                  # every gap between two tokens
            finished = [r for r in running if r[1] == 0]
            for r in finished:
                out.append((r[3] - r[2], (t - r[3]) / 255, r[5]))     # TTFT, TPOT (mean gap), worst gap
            running = [r for r in running if r[1] > 0]
            for w in done:
                waiting.remove(w)
                running.append([w[1] + 1, w[3] - 1, w[0], t, t, 0.0])   # first token comes out of prefill
    return out

def disaggregated(reqs, n_pre=1, n_dec=1, link=50e9):
    """n_pre GPUs only prefill (whole prompts; each request goes to the
    first free one), n_dec GPUs only decode (requests split round robin);
    the KV cache crosses a link (50 GB/s ~ 400 Gb/s InfiniBand)."""
    free, ready = [0.0] * n_pre, []
    for arrival, prompt, n_out in reqs:
        g = int(np.argmin(free))
        free[g] = max(free[g], arrival) + step_time(prompt, 0, 0)
        ready.append((free[g] + KV_TOKEN * prompt / link, arrival, prompt, n_out, free[g]))
    ready.sort()
    out = []
    for d in range(n_dec):
        mine, running, j, t = ready[d::n_dec], [], 0, 0.0
        while j < len(mine) or running:
            while j < len(mine) and mine[j][0] <= t:
                r = mine[j]; running.append([r[2] + 1, r[3] - 1, r[1], r[4], r[4], 0.0]); j += 1
            if not running:
                t = mine[j][0]; continue
            t += step_time(0, len(running), sum(r[0] for r in running))
            for r in running:
                r[0] += 1; r[1] -= 1
                r[5] = max(r[5], t - r[4]); r[4] = t
            for r in [r for r in running if r[1] == 0]:
                out.append((r[3] - r[2], (t - r[3]) / 255, r[5]))
            running = [r for r in running if r[1] > 0]
    return out

# Two metrics per request: TPOT, the mean gap between its output tokens,
# and its worst single gap (inter-token latency, ITL). SLO: first token
# within 2 s and every gap within 50 ms (loose) or 20 ms (tight).
SLO_TTFT, LOOSE, TIGHT = 2.0, 0.050, 0.020
def report(rate, name, res):
    a = np.array(res)
    p = lambda x, q: np.percentile(x, q)
    within = lambda limit: np.mean((a[:, 0] < SLO_TTFT) & (a[:, 2] < limit))
    print(f"{rate:5d} {name:26s} {p(a[:, 0], 50):6.2f}s {p(a[:, 0], 99):6.2f}s {p(a[:, 1], 99) * 1e3:7.1f}ms "
          f"{p(a[:, 2], 99) * 1e3:8.1f}ms {within(LOOSE):6.0%} {within(TIGHT):6.0%}")

print(f"KV cache per token {KV_TOKEN // 1024} KiB; a 2,000-token prompt's cache is {2000 * KV_TOKEN / 2**20:.0f} MiB, "
      f"{2000 * KV_TOKEN / 50e9 * 1e3:.1f} ms over a 50 GB/s link; its prefill alone takes "
      f"{step_time(2000, 0, 0) * 1e3:.0f} ms; a step stays weight-bound up to {FLOPS * 2 * N / BW / (2 * N):.0f} tokens")
head = f"{'req/s':>5s} {'design':26s} {'TTFT p50':>7s} {'p99':>6s} {'TPOT p99':>9s} {'worst gap p99':>13s} {'loose':>6s} {'tight':>6s}"
print(f"\n2 GPUs; share of requests whose first token comes within {SLO_TTFT:.0f} s and every later token within "
      f"{LOOSE * 1e3:.0f} ms (loose) or {TIGHT * 1e3:.0f} ms (tight)\n" + head)
for rate in (4, 8, 12):
    reqs = workload(rate)
    for name, res in (("colocated, whole prefill", colocated(reqs, None, 2)),
                      ("colocated, chunks of 512", colocated(reqs, 512, 2)),
                      ("colocated, chunks of 128", colocated(reqs, 128, 2)),
                      ("disaggregated 1 + 1", disaggregated(reqs, 1, 1))):
        report(rate, name, res)

# The ratio matters: 4 GPUs at 24 requests a second, colocated or split.
print("\n4 GPUs\n" + head)
reqs = workload(24, 4000)
for name, res in (("colocated, chunks of 128", colocated(reqs, 128, 4)),
                  ("disaggregated 2 + 2", disaggregated(reqs, 2, 2)),
                  ("disaggregated 3 + 1", disaggregated(reqs, 3, 1))):
    report(24, name, res)
