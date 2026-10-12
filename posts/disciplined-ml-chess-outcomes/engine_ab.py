"""A/B on the same split: the move-10 model with and without Stockfish's evaluation.

Reads features.parquet and engine.parquet (from engine.py). The evaluation
is a move-10 feature; the contract gives it its own model, win_move10_engine,
whose columns are the move-10 model's plus this one.
"""

import numpy as np

from models import bootstrap_gain, brier, ece, elo_refit, fit_predict, load, split

f = load(engine=True)
gains = []
for seed in (42, 1, 2, 3, 4):
    train, valid = split(f, seed=seed)
    va, base = fit_predict("win_move10", train, valid, seed=seed)
    _, eng = fit_predict("win_move10_engine", train, valid, seed=seed)
    y = va.score.to_numpy()
    refit = elo_refit(train, valid, "win_move10")
    gains.append((brier(refit, y) - brier(base, y), brier(refit, y) - brier(eng, y)))
    if seed == 42:
        print(f"split 42, {len(va):,} validation games: Brier Elo refit {brier(refit, y):.4f}, "
              f"move-10 model {brier(base, y):.4f}, with engine {brier(eng, y):.4f} (ECE {ece(eng, y):.3f})")
        g, lo, hi = bootstrap_gain(base, eng, y)
        print(f"engine column's gain over the same model without it: {g:+.4f} (95% CI {lo:+.4f}, {hi:+.4f})")
        even = (va.m10_material_diff == 0).to_numpy()
        for name, mask in (("material even", even), ("material uneven", ~even)):
            print(f"  {name:15s} {mask.sum():>6,} games: Brier {brier(base[mask], y[mask]):.4f} -> "
                  f"{brier(eng[mask], y[mask]):.4f}")
print(f"\n{'split seed':>10s} {'gain over Elo refit: without':>29s} {'with engine':>12s} {'ratio':>6s}")
for seed, (a, b) in zip((42, 1, 2, 3, 4), gains):
    print(f"{seed:>10} {a:>+29.4f} {b:>+12.4f} {b / a:>5.1f}x")
a, b = np.mean(gains, axis=0)
print(f"{'mean':>10s} {a:>+29.4f} {b:>+12.4f} {b / a:>5.1f}x")
