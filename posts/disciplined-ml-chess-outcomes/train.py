"""Train the four models on the 80/20 split and score them against baselines.

Reads features.parquet (from features.py). Win models are scored with the
Brier score against the Elo formula and a refitted Elo curve; the Elo model
with mean absolute error against predicting the training mean.
"""

import numpy as np

from contract import eligible
from models import bootstrap_gain, brier, ece, elo_formula, elo_refit, fit_predict, load, split

f = load()
train, valid = split(f)
print(f"train {len(train):,} games, validation {len(valid):,}; abandoned games excluded from both")

print(f"\n{'model':12s} {'games':>6s} {'Brier: mean':>11s} {'Elo':>7s} {'Elo refit':>9s} {'model':>7s}"
      f" {'gain over Elo refit (95% CI)':>29s} {'ECE':>6s}")
for m in ("win_pregame", "win_move3", "win_move10"):
    va, p = fit_predict(m, train, valid)
    y = va.score.to_numpy()
    mean = np.full(len(y), eligible(train, m).score.mean())
    refit = elo_refit(train, valid, m)
    g, lo, hi = bootstrap_gain(refit, p, y)
    print(f"{m:12s} {len(va):>6,} {brier(mean, y):>11.4f} {brier(elo_formula(va).to_numpy(), y):>7.4f} "
          f"{brier(refit, y):>9.4f} {brier(p, y):>7.4f} {g:>+12.4f} ({lo:+.4f}, {hi:+.4f}) {ece(p, y):>6.3f}")

va, p = fit_predict("elo_move10", train, valid)
truth = va[["elo_w", "elo_b"]].to_numpy()
base = eligible(train, "elo_move10")[["elo_w", "elo_b"]].mean().to_numpy()
print(f"\nelo_move10: {len(va):,} games; MAE predicting the training mean {np.abs(truth - base).mean():.1f}, "
      f"model {np.abs(truth - p).mean():.1f} rating points")
print(f"correlation of predicted and true mean rating: "
      f"{np.corrcoef(p.mean(1), truth.mean(1))[0, 1]:.2f}")
