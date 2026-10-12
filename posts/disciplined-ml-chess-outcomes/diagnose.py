"""Why the win models stop where they do: four diagnostics on the move-10 model.

Reads features.parquet. Uses the same 80/20 split as train.py.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from contract import FEATURES, eligible
from models import booster, brier, elo_refit, fit_predict, load, split

f = load()
train, valid = split(f)
m = "win_move10"
tr, va = eligible(train, m), eligible(valid, m)
y = va.score.to_numpy()
refit = elo_refit(train, valid, m)

# 1. Win rate by rating gap and material after move 10, from the data.
print("White's mean score by rating gap (rows) and material after move 10 (columns), training games")
gap = pd.cut(tr.elo_diff, [-np.inf, -150, -50, 50, 150, np.inf],
             labels=["Black +150", "Black +50", "even", "White +50", "White +150"])
mat = pd.cut(tr.m10_material_diff, [-np.inf, -2, -1, 0, 1, np.inf],
             labels=["<=-2", "-1", "0", "+1", ">=+2"])
cells = tr.assign(gap=gap, mat=mat).dropna(subset=["mat"]).pivot_table(
    index="gap", columns="mat", values="score", aggfunc=["mean", "size"], observed=False)
print(cells["mean"].round(2).to_string())
print("games per cell:")
print(cells["size"].to_string())

# 2. The same question for castling: does castling by move 10 tilt the even-rating row?
even = tr[tr.elo_diff.abs() <= 50]
print("\neven ratings (|gap| <= 50): White's mean score by who had castled by move 10")
print(even.groupby(["m10_castled_w", "m10_castled_b"]).score.agg(["mean", "size"]).round(3).to_string())

# 3. Learning curve: validation Brier as the training set grows.
print(f"\nlearning curve, {m}: validation Brier (Elo refit on the full training set: {brier(refit, y):.4f})")
cols = FEATURES[m]
print(f"  {'games':>6s}  mean of 3 subsamples (lowest, highest)")
for n in (2_500, 5_000, 10_000, 20_000, 40_000, len(tr)):
    scores = []
    for s in range(3):
        sub = tr.sample(n, random_state=s) if n < len(tr) else tr
        scores.append(brier(np.clip(booster(s).fit(sub[cols], sub.score).predict(va[cols]), 0, 1), y))
    print(f"  {n:>6,}  {np.mean(scores):.4f} ({min(scores):.4f}, {max(scores):.4f})")

# 4. Permutation importance: how much Brier worsens when one column is shuffled.
_, p = fit_predict(m, train, valid)          # the same predictions train.py scores
model = booster().fit(tr[cols], tr.score)
imp = permutation_importance(model, va[cols], y, scoring="neg_mean_squared_error",
                             n_repeats=5, random_state=0)
order = np.argsort(-imp.importances_mean)
total = imp.importances_mean.clip(min=0).sum()
print("\npermutation importance (Brier increase when shuffled; share of the total)")
for i in order[:8]:
    print(f"  {cols[i]:22s} {imp.importances_mean[i]:.5f}  {imp.importances_mean[i].clip(min=0) / total:5.1%}")

# 5. Where the error lives: Brier by rating gap and by material balance.
seg = va.assign(err=(p - y) ** 2, gap=pd.cut(va.elo_diff.abs(), [0, 50, 150, 300, np.inf], right=False))
print("\nvalidation Brier of the move-10 model, by |rating gap|")
print(seg.groupby("gap", observed=True).err.agg(["mean", "size"]).round(4).to_string())
print("by material after move 10 (even vs not)")
print(seg.groupby(seg.m10_material_diff == 0).err.agg(["mean", "size"]).round(4)
      .rename(index={True: "even", False: "uneven"}).to_string())
