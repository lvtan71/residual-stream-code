"""Does the result survive a different split? Five random splits, a split by
time, and a split that keeps every player on one side.

Reads features.parquet. For each split and win model, the Brier gain of the
booster over the refitted Elo curve; for the rating model, MAE against the
training mean.
"""

import numpy as np

from contract import eligible
from models import brier, elo_refit, fit_predict, load, split

f = load()
print(f"{'split':>12s} {'train':>7s} {'valid':>6s} {'pregame':>9s} {'move 3':>9s} {'move 10':>9s}"
      f" {'rating MAE: mean -> model':>26s}")
gains = {m: [] for m in ("win_pregame", "win_move3", "win_move10")}
for kind, seed in [("random", s) for s in (42, 1, 2, 3, 4)] + [("time", 0), ("players", 0)]:
    train, valid = split(f, seed=seed, kind=kind)
    row = []
    for m in gains:
        va, p = fit_predict(m, train, valid, seed=seed)
        y = va.score.to_numpy()
        g = brier(elo_refit(train, valid, m), y) - brier(p, y)
        if kind == "random":
            gains[m].append(g)
        row.append(f"{g:+9.4f}")
    va, p = fit_predict("elo_move10", train, valid, seed=seed)
    truth = va[["elo_w", "elo_b"]].to_numpy()
    base = eligible(train, "elo_move10")[["elo_w", "elo_b"]].mean().to_numpy()
    name = f"random {seed}" if kind == "random" else kind
    print(f"{name:>12s} {len(train):>7,} {len(valid):>6,} {' '.join(row)} "
          f"{np.abs(truth - base).mean():>17.1f} -> {np.abs(truth - p).mean():.1f}")
    if kind == "random" and seed == 4:
        print(f"{'random mean':>12s} {'':>7s} {'':>6s} " + " ".join(f"{np.mean(g):+9.4f}" for g in gains.values()))
