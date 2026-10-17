import numpy as np

# Adam's moving averages on a gradient that is always 0.001: a small, perfectly
# consistent gradient. Both averages start at 0, so both start too small, and
# v (averaged with 0.999) far more than m (0.9). Without correction, the first
# steps are too large.
b1, b2, eta, g = 0.9, 0.999, 0.01, 0.001
m = v = 0.0
for t in range(1, 6):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    raw, fixed = m / np.sqrt(v), (m / (1 - b1 ** t)) / np.sqrt(v / (1 - b2 ** t))
    print(f"t={t}: m {m:.2e}  v {v:.2e}   step without correction {eta * raw:.5f}"
          f"   with correction {eta * fixed:.5f}")

# The first corrected step is eta * g / |g| for every parameter, whatever the gradient's size.
for g in (1e-6, 1e-2, 10.0):
    m, v = (1 - b1) * g, (1 - b2) * g * g
    print(f"gradient {g:>7g}: first Adam step {eta * (m / (1 - b1)) / np.sqrt(v / (1 - b2)):.4f}"
          f"   first gradient-descent step {eta * g:.1e}")

# How long does the uncorrected step stay too large? Its ratio to the
# corrected one is (1 - b1^t) / sqrt(1 - b2^t).
t = np.arange(1, 1001)
ratio = (1 - b1 ** t) / np.sqrt(1 - b2 ** t)
print(f"uncorrected / corrected: peak {ratio.max():.2f} at t={t[ratio.argmax()]}, "
      f"{ratio[99]:.2f} at t=100, {ratio[999]:.2f} at t=1000")
