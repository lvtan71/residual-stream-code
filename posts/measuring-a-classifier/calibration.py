import numpy as np
from overfit import sigmoid, log_loss
from router_models import RUNS, pooled

def reliability(p, y, bins=5):
    """Mean predicted vs observed rate per probability bin, and the ECE."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for k in range(bins):
        m = which == k
        if m.any():
            rows.append((edges[k], edges[k + 1], m.sum(), p[m].mean(), y[m].mean()))
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return rows, ece

for name in ["no penalty", "L2 + keywords"]:
    z, y = pooled(name, "test")
    rows, ece = reliability(sigmoid(z), y)
    print(f"{name}: ECE {ece:.3f}, Brier {((sigmoid(z) - y) ** 2).mean():.3f}")
    for lo, hi, n, conf, rate in rows:
        print(f"   p in [{lo:.1f}, {hi:.1f}): {n:3d} predictions, "
              f"mean p {conf:.2f}, actually serving {rate:.2f}")

# How big is ECE for probabilities that are calibrated by construction?
p = sigmoid(pooled("L2 + keywords", "test")[0])
rng = np.random.default_rng(0)
floor = [reliability(p, (rng.random(len(p)) < p).astype(float))[1] for _ in range(500)]
print(f"ECE of perfectly calibrated labels: mean {np.mean(floor):.3f},"
      f" 95th percentile {np.percentile(floor, 95):.3f}")

# Temperature scaling: one number tau, fit on each split's validation set.
grid = np.exp(np.linspace(np.log(0.2), np.log(5), 400))
for name in ["no penalty", "L2", "L2 + keywords"]:
    before, after, taus = [], [], []
    for run in RUNS:
        zv, yv = run[name]["val"]
        tau = grid[np.argmin([log_loss(sigmoid(zv / t), yv) for t in grid])]
        zt, yt = run[name]["test"]
        before.append(sigmoid(zt)); after.append(sigmoid(zt / tau)); taus.append(tau)
    y = pooled(name, "test")[1]
    e0, e1 = reliability(np.concatenate(before), y)[1], reliability(np.concatenate(after), y)[1]
    print(f"{name:14s} tau median {np.median(taus):.2f} (range {min(taus):.2f}-{max(taus):.2f}):"
          f"  ECE {e0:.3f} -> {e1:.3f}")