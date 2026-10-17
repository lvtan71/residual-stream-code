import re
from math import comb

# From tts_hard.out's per-problem counts (c of n = 16 samples right):
# the average accuracy of one sample, and the unbiased estimate of the
# chance that at least one of k samples is right (pass@k, Chen et al.,
# 2021): 1 - C(n - c, k) / C(n, k), averaged over problems. That is what a
# perfect verifier choosing among k samples would score.
counts = [int(m) for m in re.findall(r"^problem \d+: (\d+)/16", open("tts_hard.out").read(), re.M)]
n = 16
print(f"{len(counts)} problems; one sample right on average: {sum(counts) / (n * len(counts)):.1%}")
for k in (1, 2, 4, 8, 16):
    p = sum(1 - comb(n - c, k) / comb(n, k) for c in counts) / len(counts)
    print(f"  pass@{k:<2d} (some of {k:2d} samples right): {p:.1%}")