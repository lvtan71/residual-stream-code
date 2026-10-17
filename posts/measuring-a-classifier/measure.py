import numpy as np
from overfit import sigmoid
from router_models import RUNS, pooled

def counts(p, y, t=0.5):
    yes = p >= t
    return ((yes & (y == 1)).sum(), (yes & (y == 0)).sum(),    # TP, FP
            (~yes & (y == 1)).sum(), (~yes & (y == 0)).sum())  # FN, TN

def roc_auc(p, y):
    """Probability that a random positive scores above a random negative."""
    ranks = np.empty(len(p)); ranks[np.argsort(p)] = np.arange(1, len(p) + 1)
    pos, neg = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)

def average_precision(p, y):
    """Area under the precision-recall curve (mean precision at each hit)."""
    hits = y[np.argsort(-p)]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return (precision * hits).sum() / hits.sum()

# One split, one model, the default threshold.
z, y = RUNS[0]["L2 + keywords"]["test"]
tp, fp, fn, tn = counts(sigmoid(z), y)
print(f"split 0 test: TP {tp}  FP {fp}  FN {fn}  TN {tn}")

# All three models, 800 test predictions pooled over 20 splits.
for name in ["no penalty", "L2", "L2 + keywords"]:
    z, y = pooled(name, "test")
    p = sigmoid(z)
    tp, fp, fn, tn = counts(p, y)
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"{name:14s} acc {(tp + tn) / len(y):.3f}  precision {precision:.3f}"
          f"  recall {recall:.3f}  F1 {f1:.3f}  ROC-AUC {roc_auc(p, y):.3f}"
          f"  AP {average_precision(p, y):.3f}")

# The best router's precision-recall trade-off, threshold by threshold.
z, y = pooled("L2 + keywords", "test")
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    tp, fp, fn, tn = counts(sigmoid(z), y, t)
    print(f"threshold {t:.1f}: precision {tp / (tp + fp):.3f}  recall {tp / (tp + fn):.3f}"
          f"  cost {2 * fn + fp}")

# Three ways to pick the threshold when a miss costs twice a false alarm,
# each scored on the test predictions.
grid = np.round(np.arange(0.02, 0.985, 0.01), 2)
cost = lambda p, y, t: 2 * ((p < t) & (y == 1)).sum() + ((p >= t) & (y == 0)).sum()
from_theory = from_validation = 0
for run in RUNS:
    pv, yv = sigmoid(run["L2 + keywords"]["val"][0]), run["L2 + keywords"]["val"][1]
    pt, yt = sigmoid(run["L2 + keywords"]["test"][0]), run["L2 + keywords"]["test"][1]
    t_val = grid[int(np.argmin([cost(pv, yv, t) for t in grid]))]
    from_validation += cost(pt, yt, t_val)
    from_theory += cost(pt, yt, 1 / 3)
in_hindsight = min(cost(sigmoid(z), y, t) for t in grid)
print(f"test cost at 0.5: {cost(sigmoid(z), y, 0.5)};  theory's 1/3: {from_theory};"
      f"  tuned on validation: {from_validation};  tuned on test itself: {in_hindsight}")