import math

# Softmax dilution: one relevant key scores delta above n - 1 others that all
# score the same. Its share of the attention is e^delta / (e^delta + n - 1).
for delta in (4, 8):
    shares = ", ".join(f"{n:>7,} keys {math.exp(delta) / (math.exp(delta) + n - 1):6.1%}" for n in (100, 1_000, 10_000, 100_000))
    print(f"score margin {delta}: {shares}")
print(f"margin needed for half the attention among 100,000 keys: ln(99,999) = {math.log(99_999):.1f}")
