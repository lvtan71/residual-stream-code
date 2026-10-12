"""Does the result survive a different split? Five random 80/20 splits.

Reads features.parquet. For each split and win model, the Brier gain of the
booster over the refitted Elo curve; for the Elo model, MAE against the mean.
"""

import numpy as np

from contract import eligible
from models import brier, elo_refit, fit_predict, load, split

f = load()
print(f"{'split seed':>10s} {'pregame':>9s} {'move 3':>9s} {'move 10':>9s} {'Elo MAE: mean -> model':>24s}")
gains = {m: [] for m in ("win_pregame", "win_move3", "win_move10")}
for seed in (42, 1, 2, 3, 4):
    train, valid = split(f, seed=seed)
    row = []
    for m in gains:
        va, p = fit_predict(m, train, valid, seed=seed)
        y = va.score.to_numpy()
        gains[m].append(brier(elo_refit(train, valid, m), y) - brier(p, y))
        row.append(f"{gains[m][-1]:+9.4f}")
    va, p = fit_predict("elo_move10", train, valid, seed=seed)
    truth = va[["elo_w", "elo_b"]].to_numpy()
    base = eligible(train, "elo_move10")[["elo_w", "elo_b"]].mean().to_numpy()
    print(f"{seed:>10} {' '.join(row)} {np.abs(truth - base).mean():>15.1f} -> {np.abs(truth - p).mean():.1f}")
print(f"{'mean':>10s} " + " ".join(f"{np.mean(g):+9.4f}" for g in gains.values()))
