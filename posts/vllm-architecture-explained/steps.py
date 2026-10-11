import time
import numpy as np
import torch
from transformers import DynamicCache
from lm import model, tok, DEVICE

# What one forward pass of Qwen2.5-1.5B costs on this laptop's GPU. Two
# kinds of step:
#   decode:  B sequences, one new token each, each over C cached tokens;
#   prefill: one sequence, T new prompt tokens, nothing cached before.
# Prefill computes logits only for the last position, as a server does.
# Then a cost model a scheduler can use for any mix of the two:
#   time = a + b * (new tokens) + c * (cached tokens read by decodes) + d * T^2
# (the last term is a prompt's attention over itself), fitted by least squares.
blog = tok(open("blog_train.txt").read()).input_ids
sync = torch.mps.synchronize if DEVICE == "mps" else torch.cuda.synchronize if DEVICE == "cuda" else (lambda: None)

def timed(fn, trials=15):
    fn(); sync()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter(); fn(); sync()
        times.append(time.perf_counter() - t0)
    return sorted(times)[trials // 2]

@torch.no_grad()
def decode_step(B, C):
    ids = torch.tensor([blog[i * C:(i + 1) * C] for i in range(B)], device=DEVICE)
    cache = DynamicCache()
    model.model(ids, past_key_values=cache, use_cache=True)          # fill the cache, no logits
    def step():
        model(ids[:, -1:], past_key_values=cache, use_cache=True)
        cache.crop(-1)                                               # keep the context fixed
    return timed(step)

@torch.no_grad()
def prefill_step(T):
    ids = torch.tensor([blog[:T]], device=DEVICE)
    def step():
        h = model.model(ids, past_key_values=DynamicCache(), use_cache=True).last_hidden_state
        model.lm_head(h[:, -1:])
    return timed(step)

rows = []                                                            # (new tokens, cached read, T^2, seconds)
print("decode, ms per step (tokens/s)")
print("   B   " + "".join(f"{f'C = {C}':>20s}" for C in (128, 512, 2048)))
for B in (1, 4, 16, 64, 128):
    line = f"{B:4d}   "
    for C in (128, 512, 2048):
        if B * C > 128 * 512:                                        # keep the cache under ~2 GB
            line += f"{'':>20s}"
            continue
        t = decode_step(B, C)
        rows.append((B, B * C, 0, t))
        line += f"{t * 1e3:10.1f} ({B / t:5.0f}/s)"
    print(line)
print("prefill, one sequence")
for T in (16, 64, 128, 256, 512, 1024, 2048, 4096):
    t = prefill_step(T)
    rows.append((T, 0, T * T, t))
    print(f"  T = {T:4d}: {t * 1e3:7.1f} ms ({T / t:5.0f} tokens/s)")

X = np.array([[1, n, c, q] for n, c, q, _ in rows], float)
y = np.array([t for *_, t in rows])
coef = np.linalg.lstsq(X / y[:, None], np.ones(len(y)), rcond=None)[0]   # fit relative error
err = np.abs(X @ coef - y) / y
print(f"fit: {coef[0] * 1e3:.1f} ms + {coef[1] * 1e6:.0f} us per new token + {coef[2] * 1e9:.0f} ns per cached "
      f"token read + {coef[3] * 1e12:.1f} ps x T^2; error over {len(y)} steps: mean {err.mean():.1%}, max {err.max():.1%}")
np.save("cost.npy", coef)
