import numpy as np
from router_data import split
from softmax_basics import softmax, cross_entropy
from router4 import four_way, train

# 1. Is 4,000 steps enough, or is it quietly early-stopping some models?
# 2. What does each model look like on its own training data?
CONFIGS = {"L2 0.003": (0.003, 0.0), "no penalty": (0.0, 0.0),
           "smoothing 0.1": (0.0, 0.1), "smoothing 0.1 + L2 0.003": (0.003, 0.1)}
rows = {}
for seed in range(20):
    ((Xtr, ytr), (Xva, yva), _), _ = four_way(*split(seed))
    for name, (lam, smooth) in CONFIGS.items():
        for steps in (4000, 16000):
            W, b = train(Xtr, ytr, lam, smooth, steps=steps)
            P = softmax(Xva @ W + b)
            rows.setdefault((name, steps), []).append((
                cross_entropy(Xva @ W + b, yva), np.linalg.norm(W),
                cross_entropy(Xtr @ W + b, ytr),              # hard labels, no penalty
                P.max(1).mean(), (P.argmax(1) == yva).mean()))
corrected = np.sqrt(1 / 20 + 40 / 120)
base = {s: np.array(rows[("L2 0.003", s)])[:, 0] for s in (4000, 16000)}
print(f"{'':25s} {'steps':>6s} {'val CE':>7s} {'vs L2':>15s} {'|W|':>6s} {'train CE':>9s} "
      f"{'mean top p':>11s} {'val acc':>8s}")
for (name, steps), r in rows.items():
    r = np.array(r)
    d = r[:, 0] - base[steps]
    vs = f"{d.mean():+.3f} ± {d.std(ddof=1) * corrected:.3f}" if d.any() else ""
    print(f"{name:25s} {steps:6,d} {r[:, 0].mean():7.3f} {vs:>15s} {r[:, 1].mean():6.1f} "
          f"{r[:, 2].mean():9.3f} {r[:, 3].mean():11.3f} {r[:, 4].mean():8.3f}")