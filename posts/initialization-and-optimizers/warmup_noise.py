import numpy as np

# 100,000 parameters whose gradients are pure noise of size 1: no direction is
# right, so ideally they shouldn't move much. Two things about Adam's early steps:
#   the typical step, m_hat / sqrt(v_hat), in units of eta;
#   the per-parameter rate 1 / sqrt(v_hat), whose true value here is 1. With few
#   samples, v_hat can be tiny by chance, and then that parameter's rate is huge.
rng = np.random.default_rng(0)
b1, b2, n = 0.9, 0.999, 100_000
m, v = np.zeros(n), np.zeros(n)
for t in range(1, 5001):
    g = rng.normal(size=n)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
    if t in (1, 2, 5, 10, 100, 1000, 5000):
        step = m_hat / np.sqrt(v_hat)
        print(f"step {t:5d}: typical update {np.sqrt((step ** 2).mean()):.3f} x eta;  "
              f"parameters with a rate over 3x its true value: {(1 / np.sqrt(v_hat) > 3).mean():6.2%}")
print(f"long-run update size, sqrt((1 - b1) / (1 + b1)) = {np.sqrt((1 - b1) / (1 + b1)):.3f}")
