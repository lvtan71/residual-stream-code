"""Shared pieces: the split, the model, the baselines and the metrics."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from contract import CATEGORICAL, FEATURES, TARGET, eligible


def load():
    f = pd.read_parquet("features.parquet")
    for c in CATEGORICAL:
        f[c] = f[c].astype("category")
    return f


def split(f, seed=42, valid=0.2):
    """A random 80/20 split of games, made once, before eligibility is applied."""
    is_valid = np.random.default_rng(seed).random(len(f)) < valid
    return f[~is_valid], f[is_valid]


def booster(seed=0):
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, early_stopping=True,
                                         validation_fraction=0.1, n_iter_no_change=20,
                                         categorical_features="from_dtype", random_state=seed)


def fit_predict(model, train, valid, seed=0, cols=None):
    """Fit one booster per target column on eligible training games; predict eligible validation games."""
    tr, va = eligible(train, model), eligible(valid, model)
    cols = cols or FEATURES[model]
    targets = TARGET[model] if isinstance(TARGET[model], list) else [TARGET[model]]
    preds = np.column_stack([booster(seed).fit(tr[cols], tr[t]).predict(va[cols]) for t in targets])
    return va, preds if len(targets) > 1 else np.clip(preds[:, 0], 0, 1)


def elo_formula(df):
    """Expected score for White from the rating difference alone, no parameters."""
    return 1 / (1 + 10 ** (-df.elo_diff / 400))


def elo_refit(train, valid, model):
    """The same curve with its scale and White's edge fitted: logistic regression on elo_diff."""
    tr, va = eligible(train, model), eligible(valid, model)
    w = np.column_stack([tr.score, 1 - tr.score]).ravel()           # a draw counts half a win
    x = np.repeat(tr[["elo_diff"]].to_numpy() / 400, 2, axis=0)
    y = np.tile([1, 0], len(tr))
    lr = LogisticRegression().fit(x, y, sample_weight=w)
    return lr.predict_proba(va[["elo_diff"]].to_numpy() / 400)[:, 1]


def brier(p, y):
    return np.mean((p - y) ** 2)


def ece(p, y, bins=10):
    """Expected calibration error: the size-weighted gap between predicted and actual score per bin."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def bootstrap_gain(p_base, p_model, y, n=1000, seed=0):
    """95% interval for Brier(base) - Brier(model), resampling validation games."""
    rng = np.random.default_rng(seed)
    d = (p_base - y) ** 2 - (p_model - y) ** 2
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)
