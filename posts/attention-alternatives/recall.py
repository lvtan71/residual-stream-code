"""Associative recall: write n key-value pairs, then ask for each value back.

Three memories, no training:
  cache   keeps every key and value, and reads with softmax attention
  linear  adds each pair into one d x d state, S += v k^T (linear attention)
  delta   erases what the state holds at k, then writes, S += (v - S k) k^T

A read counts as correct when the value it returns is closer to the right
value than to any other stored value.
"""

import numpy as np

rng = np.random.default_rng(0)


def pairs(n, d):
    keys = rng.standard_normal((n, d))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)    # unit keys
    values = rng.standard_normal((n, d))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return keys, values


def read_cache(keys, values, queries, scale=16.0):
    w = np.exp(scale * queries @ keys.T)                     # softmax over every stored key
    return (w / w.sum(1, keepdims=True)) @ values


def write_linear(keys, values):
    return values.T @ keys                                   # sum of outer products v k^T


def write_delta(keys, values, beta=1.0):
    S = np.zeros((values.shape[1], keys.shape[1]))
    for k, v in zip(keys, values):
        S += beta * np.outer(v - S @ k, k)                   # replace what k reads, keep the rest
    return S


def score(out, values):
    cos = (out / np.linalg.norm(out, axis=1, keepdims=True)) @ values.T
    right = cos.argmax(1) == np.arange(len(values))
    return right.mean(), np.diag(cos).mean()


def recall(n, d):
    keys, values = pairs(n, d)
    return {
        "cache": score(read_cache(keys, values, keys), values),
        "linear": score(keys @ write_linear(keys, values).T, values),
        "delta": score(keys @ write_delta(keys, values).T, values),
    }


d = 64
print(f"key and value width d = {d}; the state is d x d = {d * d:,} numbers")
print(f"{'pairs':>6s} {'cache size':>11s} {'cache':>12s} {'linear':>12s} {'delta':>12s}")
print(f"{'':>6s} {'(numbers)':>11s}" + "  correct cos" * 3)
for n in (16, 32, 64, 128, 256, 512, 1024, 4096):
    r = recall(n, d)
    print(f"{n:>6,} {2 * n * d:>11,}" + "".join(f" {r[m][0]:6.0%} {r[m][1]:5.2f}" for m in r))

n = 256
print(f"\nstate size dial: {n} pairs, linear and delta memories of width d")
print(f"{'d':>5s} {'state':>8s} {'linear':>12s} {'delta':>12s}")
for d in (32, 64, 128, 256, 512):
    r = recall(n, d)
    print(f"{d:>5} {d * d:>8,}" + "".join(f" {r[m][0]:6.0%} {r[m][1]:5.2f}" for m in ("linear", "delta")))

# Which pairs survive: 256 pairs in a width-64 state, scored by when they were written.
d, n = 64, 256
keys, values = pairs(n, d)
print(f"\n{n} pairs, d = {d}: share read back correctly, by when the pair was written")
print(f"{'written':>14s} {'linear':>7s} {'delta':>7s}")
reads = {"linear": keys @ write_linear(keys, values).T, "delta": keys @ write_delta(keys, values).T}
cos = {m: (o / np.linalg.norm(o, axis=1, keepdims=True)) @ values.T for m, o in reads.items()}
for lo in range(0, n, n // 8):
    hi = lo + n // 8
    row = [(cos[m][lo:hi].argmax(1) == np.arange(lo, hi)).mean() for m in reads]
    print(f"{f'{lo + 1}-{hi}':>14s}" + "".join(f" {x:7.0%}" for x in row))

# Overwriting: the same key is written twice, first with value a, later with b.
d = 64
keys, values = pairs(32, d)
a, b = values[0], rng.standard_normal(d)
b /= np.linalg.norm(b)
keys2 = np.vstack([keys, keys[:1]])
values2 = np.vstack([values, b[None]])
print("\nkey 0 written with value a, 31 other pairs, then key 0 again with value b")
for name, S in (("linear", write_linear(keys2, values2)), ("delta", write_delta(keys2, values2))):
    out = S @ keys[0]
    out /= np.linalg.norm(out)
    print(f"  {name:6s} reads key 0: cos to a {out @ a:5.2f}, cos to b {out @ b:5.2f}")
