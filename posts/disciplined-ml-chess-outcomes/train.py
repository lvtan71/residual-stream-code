"""Train the models on one random 80/20 split and score them against baselines.

Reads features.parquet (from features.py). Win models are scored with the
Brier score against the Elo formula and a refitted Elo curve; the rating
models with mean absolute error against predicting the training mean.
"""

import numpy as np

from contract import eligible
from models import bootstrap_gain, brier, ece, elo_formula, elo_refit, fit_predict, load, split

f = load()
train, valid = split(f)
print(f"split: {len(train):,} training and {len(valid):,} validation games; each model then keeps the "
      f"played games that reach its horizon")

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

print("\nratings from play: mean absolute error in rating points, both players")
for m in ("elo_move10", "elo_time_only", "elo_moves_only"):
    va, p = fit_predict(m, train, valid)
    truth = va[["elo_w", "elo_b"]].to_numpy()
    if m == "elo_move10":
        base = eligible(train, m)[["elo_w", "elo_b"]].mean().to_numpy()
        print(f"  {'training mean':24s} {np.abs(truth - base).mean():6.1f}   ({len(va):,} games)")
    print(f"  {m:24s} {np.abs(truth - p).mean():6.1f}   correlation of the pair's mean rating "
          f"{np.corrcoef(p.mean(1), truth.mean(1))[0, 1]:.2f}")
