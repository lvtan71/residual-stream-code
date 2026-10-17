import numpy as np
from router_data import split, ROUTES
from softmax_basics import cross_entropy, softmax
from router4 import four_way, train, top_label_ece

CONFIGS = {"L2 0.003": (0.003, 0.0), "no penalty": (0.0, 0.0),
           "smoothing 0.1": (0.0, 0.1), "smoothing 0.1 + L2 0.003": (0.003, 0.1)}
val, test = {c: [] for c in CONFIGS}, {c: [] for c in CONFIGS}
for seed in range(20):
    ((Xtr, ytr), (Xva, yva), (Xte, yte)), _ = four_way(*split(seed))
    for name, (lam, smooth) in CONFIGS.items():
        W, b = train(Xtr, ytr, lam, smooth)
        P = softmax(Xva @ W + b)
        val[name].append((cross_entropy(Xva @ W + b, yva), (P.argmax(1) == yva).mean(),
                          top_label_ece(P, yva)))
        test[name].append((softmax(Xte @ W + b).argmax(1), yte, cross_entropy(Xte @ W + b, yte)))

base = np.array(val["L2 0.003"])
corrected = np.sqrt(1 / 20 + 40 / 120)                  # Nadeau-Bengio, "Features and Overfitting"
vs = lambda d: f"{d.mean():+.3f} ± {d.std(ddof=1) * corrected:.3f}" if d.any() else ""
print(f"{'':25s} {'val CE':>7s} {'vs L2':>15s} {'val acc':>8s} {'vs L2':>15s} {'ECE':>6s} {'vs L2':>15s}")
for name, rows in val.items():
    r = np.array(rows)
    d = r - base                                         # paired, split by split
    print(f"{name:25s} {r[:, 0].mean():7.3f} {vs(d[:, 0]):>15s} {r[:, 1].mean():8.3f} "
          f"{vs(d[:, 1]):>15s} {r[:, 2].mean():6.3f} {vs(d[:, 2]):>15s}")
both, smooth = np.array(val["smoothing 0.1 + L2 0.003"]), np.array(val["smoothing 0.1"])
print(f"smoothing + L2 vs smoothing alone: val CE {vs(both[:, 0] - smooth[:, 0])}")

best = min(val, key=lambda c: np.mean([row[0] for row in val[c]]))   # chosen on validation
conf = np.zeros((4, 4), dtype=int)
for pred, true, _ in test[best]:
    for t, p in zip(true, pred):
        conf[t, p] += 1
test_ce = np.mean([ce for *_, ce in test[best]])
print(f"\nchosen on validation CE: {best}. test CE {test_ce:.3f} (perplexity {np.exp(test_ce):.2f})")
print("test confusion, pooled over the 20 splits (rows = true route):")
print(f"{'':14s}" + "".join(f"{r:>13s}" for r in ROUTES))
for i, r in enumerate(ROUTES):
    print(f"{r:14s}" + "".join(f"{v:13d}" for v in conf[i]))
precision, recall = conf.diagonal() / conf.sum(0), conf.diagonal() / conf.sum(1)
f1 = 2 * precision * recall / (precision + recall)
for r, p, q, f in zip(ROUTES, precision, recall, f1):
    print(f"{r:14s} precision {p:.3f}  recall {q:.3f}  F1 {f:.3f}")
print(f"test accuracy {conf.trace() / conf.sum():.4f}; mean recall {recall.mean():.4f}; "
      f"macro-F1 {f1.mean():.4f}")