# What an agent's transcript costs. Every turn resends everything so far: a
# P-token prefix (system prompt and tools), then g new tokens per turn (the
# model's call plus the tool's result). Turn t reads P + (t - 1) g tokens of
# context, so T turns read T P + g T (T - 1) / 2 in all: quadratic in T.
# With prompt caching ("Prompt Caching"), turn 1 writes the prefix to the cache at
# 1.25x the base input price; each later turn reads everything sent before
# at 0.1x and writes the previous turn's new tokens at 1.25x.
P, g, price = 5_000, 1_500, 3 / 1e6           # tokens; dollars per input token (an illustrative $3 per million)
print(f"prefix {P:,} tokens, {g:,} new per turn, ${price * 1e6:.0f} per million input tokens")
print(f"{'turns':>6s} {'final context':>14s} {'tokens read in all':>19s} {'uncached':>9s} {'cached':>8s}")
for T in (5, 10, 20, 50):
    total = T * P + g * T * (T - 1) // 2
    cached = P * 1.25 + sum((P + (t - 2) * g) * 0.1 + g * 1.25 for t in range(2, T + 1))   # reread, plus the new turn written
    print(f"{T:6d} {P + (T - 1) * g:12,d}   {total:17,d}   ${total * price:7.2f}  ${cached * price:6.2f}")
