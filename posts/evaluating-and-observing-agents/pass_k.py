from math import comb

# pass@k: at least one of k attempts succeeds (what a benchmark with
# retries, or a human picking the best of k, cares about). pass^k: all k
# succeed (what a user who asks k times, or k customers, cares about).
# For one task with per-attempt success p: 1 - (1 - p)^k and p^k.
print(f"{'p':>5s} " + " ".join(f"{'pass@' + str(k):>8s}" for k in (1, 4, 8)) + "   " +
      " ".join(f"{'pass^' + str(k):>8s}" for k in (1, 4, 8)))
for p in (0.5, 0.8, 0.9, 0.95, 0.99):
    print(f"{p:5.2f} " + " ".join(f"{1 - (1 - p) ** k:8.3f}" for k in (1, 4, 8)) + "   " +
          " ".join(f"{p ** k:8.3f}" for k in (1, 4, 8)))

# From n attempts with c successes, unbiased estimates for k <= n
# (Chen et al. 2021 for pass@k; the same counting gives pass^k).
def pass_at(n, c, k): return 1 - comb(n - c, k) / comb(n, k)
def pass_hat(n, c, k): return comb(c, k) / comb(n, k)
print(f"\n5 attempts, 3 succeed: pass@1 {pass_at(5, 3, 1):.2f}, pass@3 {pass_at(5, 3, 3):.2f}, "
      f"pass^3 {pass_hat(5, 3, 3):.2f}")