"""The two bills of softmax attention, and what a fixed-size state costs instead.

Pure arithmetic: FLOPs per head at prefill, the KV cache of a
Llama-3-8B-shaped model, and the d x d state that replaces it in linear
attention.
"""

D = 128                                    # head dimension
LAYERS, KV_HEADS, BYTES = 32, 8, 2         # Llama-3-8B shape, BF16
LENGTHS = (4_096, 32_768, 131_072, 1_048_576)


def prefill_flops(n, d=D):
    """QK^T and (scores)V, each 2*n*n*d FLOPs (a multiply and an add per term)."""
    return 2 * n * n * d + 2 * n * n * d


def kv_cache_bytes(n):
    """A key and a value per token, per layer, per KV head."""
    return 2 * LAYERS * KV_HEADS * D * BYTES * n


def state_bytes():
    """Linear attention: one d x d state per KV head per layer, whatever n is."""
    return LAYERS * KV_HEADS * D * D * BYTES


print(f"prefill FLOPs per head, d = {D}")
for n in LENGTHS[:3]:
    print(f"  n = {n:>9,}  {prefill_flops(n) / 1e9:>10,.1f} GFLOPs")
r = LENGTHS[2] // LENGTHS[0]
print(f"  {r}x the tokens costs {prefill_flops(LENGTHS[2]) / prefill_flops(LENGTHS[0]):,.0f}x the work")

print(f"\nKV cache per token: {kv_cache_bytes(1) / 1024:.0f} KiB")
print(f"{'context':>11s} {'KV cache':>10s} {'linear-attention state':>23s}")
for n in LENGTHS:
    print(f"{n:>11,} {kv_cache_bytes(n) / 1e9:>7.2f} GB {state_bytes() / 1e6:>20.1f} MB")
print(f"20 sequences of {LENGTHS[2]:,} tokens: {20 * kv_cache_bytes(LENGTHS[2]) / 1e9:.0f} GB "
      f"= {20 * kv_cache_bytes(LENGTHS[2]) / 80e9:.1f} H100s of 80 GB")

n = 32_768
print(f"\nscore matrix at n = {n:,}: {n * n * BYTES / 1e9:.1f} GB per head in BF16")

print("\nlinear attention: (QK^T)V costs 4n^2d, Q(K^T V) costs 4nd^2, ratio n/d")
for n in (64, 128, 4_096, 131_072):
    print(f"  n = {n:>7,}  {4 * n * n * D / (4 * n * D * D):>8,.1f}x")

W = 4_096
print(f"\nsliding window W = {W:,}: cache capped at W tokens")
n = 32_768
pairs = sum(min(i, W) for i in range(1, n + 1))
print(f"  at n = {n:,}: cache {n // W}x smaller, {pairs / (n * (n + 1) / 2):.0%} of the causal "
      f"pairs computed; {LAYERS} layers reach {LAYERS * W:,} tokens")
