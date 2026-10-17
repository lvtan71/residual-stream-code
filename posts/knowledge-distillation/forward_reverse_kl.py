import numpy as np

# Forward vs reverse KL. The teacher p is bimodal: two bumps at -2 and +2.
# The student q is a single Gaussian, too simple to match it. Fit q's mean
# and width by minimizing each direction of KL over a grid of candidates.
x = np.linspace(-8, 8, 4001)
dx = x[1] - x[0]
normal = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
p = 0.5 * normal(-2, 0.6) + 0.5 * normal(2, 0.6)

def kl(a, b):
    """KL(a || b) = sum a log(a / b), on the grid."""
    mask = a > 1e-300
    return float(np.sum(a[mask] * (np.log(a[mask]) - np.log(b[mask] + 1e-300))) * dx)

best = {}
for m in np.linspace(-3, 3, 121):
    for s in np.linspace(0.3, 4, 75):
        q = normal(m, s)
        for name, value in (("forward KL(p || q)", kl(p, q)), ("reverse KL(q || p)", kl(q, p))):
            if name not in best or value < best[name][0]:
                best[name] = (value, m, s)
for name, (value, m, s) in best.items():
    q = normal(m, s)
    between = q[np.abs(x) < 0.5].sum() * dx                 # mass q puts in the gap between the bumps
    print(f"{name}: best q has mean {m:+.2f}, width {s:.2f}; mass in the gap |x| < 0.5: {between:.0%} "
          f"(teacher: {p[np.abs(x) < 0.5].sum() * dx:.0%})")