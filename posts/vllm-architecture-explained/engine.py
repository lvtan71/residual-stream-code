import json
import random
import numpy as np

# A serving engine's scheduler, simulated: requests with real ShareGPT
# lengths (from "Paged Attention") arrive at random (Poisson) times, and
# every step costs what the fitted model of steps.py says that mix of tokens
# costs on this laptop.
# Four ways to schedule:
#   static:   take up to `batch` waiting requests, prefill them, decode until
#             every one is done; finished requests hold their slot until then
#             (memory is not limited for it, and prompts aren't padded);
#   prefill-first: continuous batching, as vLLM's original engine did by
#             default: a step either prefills newly admitted prompts whole,
#             or decodes every running request;
#   chunked:  vLLM V1: a step has a token budget; every running decode gets
#             its token first, then prompts fill what is left, in chunks.
# Memory is a pool of 16-token blocks; when it runs out mid-answer, the
# newest request is preempted: its blocks are freed and it goes to the front
# of the queue, to recompute its prompt and answer so far later. A step's cost
# is the fitted model's: decode and prefill tokens are assumed to add, and a
# prompt chunk reads the keys and values of the prompt before it.
a, b, c, d = np.load("cost.npy")
def cost(new, cached, prefills):              # prefills: (chunk, tokens before it) pairs
    return a + b * new + c * cached + d * sum(n * n + 2 * n * p for n, p in prefills)

class Req:
    def __init__(self, i, t, prompt, answer):
        self.i, self.arrive, self.prompt, self.answer = i, t, prompt, answer
        self.done = 0                         # tokens whose keys and values are computed
        self.out = []                         # times its answer tokens came out
        self.blocks = 0

    def known(self):                          # prompt plus answer so far
        return self.prompt + len(self.out)

    def decoding(self):                       # all but the newest token computed: feed it, get the next
        return self.out and self.done == self.known() - 1

def simulate(mode, rate, n=2000, budget=2048, pool=4096, max_seqs=128, batch=16, seed=0):
    rng = random.Random(seed)
    lengths = json.load(open("sharegpt_lengths.json"))   # written by waste.py in "Paged Attention"
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate)
        reqs.append(Req(i, t, *lengths[i]))
    waiting, running, finished = [], [], []
    clock, free, preempted, pending = 0.0, pool, 0, list(reqs)

    def grow(r, upto):                        # blocks for `upto` tokens; False if the pool is empty
        nonlocal free
        need = -(-upto // 16) - r.blocks
        if need > free:
            return False
        free -= need; r.blocks += need
        return True

    while len(finished) < n:
        while pending and pending[0].arrive <= clock:
            waiting.append(pending.pop(0))
        if not waiting and not running:
            clock = pending[0].arrive
            continue
        decodes, prefills = [], []            # requests decoding one token; (request, chunk) to prefill
        if mode == "static":
            if not running:
                running = waiting[:batch]; del waiting[:batch]
                prefills = [(r, r.prompt) for r in running]
            else:
                decodes = running                         # finished ones still take a slot
        else:
            prefill_step = mode == "prefill-first" and bool(waiting)
            if prefill_step:                              # a step that only prefills new prompts
                left = budget
            else:
                for r in list(running):                   # running requests' next token first
                    if r in running and r.decoding():
                        while r in running and not grow(r, r.done + 1):
                            victim = running.pop()        # out of memory: preempt the newest
                            free += victim.blocks; victim.blocks = victim.done = 0
                            waiting.insert(0, victim); preempted += 1
                        if r in running:
                            decodes.append(r)
                left = budget - len(decodes)
                for r in running:                         # prompts still being prefilled, in chunks
                    if not r.decoding() and left > 0:
                        k = min(r.known() - r.done, left)
                        if grow(r, r.done + k):
                            prefills.append((r, k)); left -= k
            if mode == "chunked" or not decodes:          # admit waiting requests while there is room
                while waiting and left > 0 and len(running) < max_seqs:
                    r = waiting[0]
                    k = r.known() - r.done                # after a preemption: prompt and answer so far
                    if mode == "chunked":
                        k = min(k, left)
                    elif k > left:
                        break
                    if not grow(r, r.done + k):
                        break
                    waiting.pop(0); running.append(r)
                    prefills.append((r, k)); left -= k
            if prefill_step and not prefills:             # nothing could be admitted: decode instead
                for r in running:
                    if r.decoding() and grow(r, r.done + 1):
                        decodes.append(r)
        new = len(decodes) + sum(k for _, k in prefills)
        cached = sum(r.done for r in decodes) + sum(r.done for r, _ in prefills)
        clock += cost(new, cached, [(k, r.done) for r, k in prefills])
        for r in decodes:
            if len(r.out) < r.answer:
                r.done += 1; r.out.append(clock)
        for r, k in prefills:
            r.done += k
            if r.done == r.known():                       # the prompt's last chunk gives a token
                r.out.append(clock)
        for r in list(running):
            if len(r.out) >= r.answer and mode != "static":
                running.remove(r); free += r.blocks; r.blocks = 0; finished.append(r)
        if mode == "static" and all(len(r.out) >= r.answer for r in running):
            finished += running; running = []
    assert sorted(r.i for r in finished) == list(range(n)) and all(len(r.out) == r.answer for r in finished)
    ttft = [r.out[0] - r.arrive for r in finished]
    gaps = [g for r in finished for g in np.diff(r.out)]
    span = max(r.out[-1] for r in finished) - min(r.arrive for r in finished)
    toks = sum(r.answer for r in finished)
    return dict(tput=toks / span, ttft50=np.median(ttft), ttft99=np.percentile(ttft, 99),
                tpot=np.mean(gaps), gap99=np.percentile(gaps, 99), gapmax=max(gaps), pre=preempted)

if __name__ == "__main__":
    print(f"{'':26s}answer tok/s  TTFT p50  TTFT p99  time/token  p99 gap  longest gap")
    for rate in (0.5, 1.0, 1.5):
        print(f"{rate} requests a second")
        for name, mode, kw in (("static, 16", "static", dict(batch=16)), ("static, 128", "static", dict(batch=128)),
                               ("prefill-first", "prefill-first", dict(budget=4096)),
                               ("chunked, 2048", "chunked", dict(budget=2048)),
                               ("chunked, 512", "chunked", dict(budget=512))):
            m = simulate(mode, rate, **kw)
            print(f"  {name:24s}{m['tput']:9.0f}  {m['ttft50']:7.1f} s {m['ttft99']:7.1f} s  "
                  f"{m['tpot'] * 1e3:6.0f} ms  {m['gap99']:5.2f} s  {m['gapmax']:7.2f} s")
    print("chunked, 512, 1.0 request a second, smaller KV pools")
    for pool in (4096, 512, 256, 128):
        m = simulate("chunked", 1.0, budget=512, pool=pool)
        print(f"  {pool:4d} blocks ({pool * 16:6,d} tokens): {m['tput']:4.0f} tok/s, TTFT p99 {m['ttft99']:6.1f} s, "
              f"longest gap {m['gapmax']:5.2f} s, {m['pre']:4d} preemptions")
