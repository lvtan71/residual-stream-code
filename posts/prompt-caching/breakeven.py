# When prompt caching pays, from the published multipliers (Anthropic's
# prompt-caching docs, October 2026): a cache write costs 1.25x the base
# input price with a 5-minute lifetime, 2x with a 1-hour lifetime; a cache
# read costs 0.1x (0.05x for Claude Opus 5.5, 0.025x for Claude Fable 5.1).
# A prefix of P tokens sent N times within the lifetime:
#   uncached: N * P;   cached: write * P + (N - 1) * read * P.
def cost(n, write, read=0.1):
    return write + (n - 1) * read                    # in units of P tokens at the base price

for name, write in (("5-minute", 1.25), ("1-hour", 2.0)):
    n = next(n for n in range(1, 100) if cost(n, write) < n)
    print(f"{name} cache (write {write}x, read 0.1x): cheaper from use {n} on; "
          f"at 10 uses it costs {cost(10, write) / 10:.0%} of uncached, at 100 uses {cost(100, write) / 100:.0%}")

# A 20,000-token system prompt for Claude Opus 5.5 (listed: $4 per million
# input tokens, $5 for 5-minute cache writes, $0.20 for cache reads), 1,000
# requests, each within 5 minutes of the previous one, so the cache stays
# warm and is written once.
P, price, N = 20_000, 4 / 1e6, 1_000
print(f"\n20k-token prefix, {N:,} requests: uncached ${N * P * price:,.2f}, "
      f"cached ${cost(N, 1.25, read=0.05) * P * price:,.2f}")

# What the provider holds for it: the prefix's KV cache, per cached copy.
for model, kv in (("Qwen2.5-1.5B", 28 * 1024), ("Qwen2.5-32B", 256 * 1024), ("DeepSeek-R1 (MLA)", 61 * 576 * 2)):
    print(f"KV cache for that 20k-token prefix, {model:18s}: {P * kv / 2**30:5.2f} GiB")