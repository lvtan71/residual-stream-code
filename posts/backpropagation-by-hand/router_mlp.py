import numpy as np
from mlp import sigmoid, forward, backward, train
from overfit import log_loss, train as train_linear
from router_data import split, featurize
from router_models import keyword_count

def features(rs, X):
    return np.c_[X, keyword_count(rs)]            # bag of words + keyword count

models = {"linear, L2 0.003": None, "linear, L2 0.01": None}
models.update({f"16 hidden, L2 {lam}": lam for lam in [0.0, 0.003, 0.01, 0.03, 0.1]})
val, test, train_loss = {m: [] for m in models}, {m: [] for m in models}, {m: [] for m in models}
for seed in range(20):
    tr, va, te = split(seed)
    ((Xtr, ytr), (Xva, yva), (Xte, yte)), _ = featurize(tr, va, te)
    A, B, C = features(tr, Xtr), features(va, Xva), features(te, Xte)
    for name, lam in models.items():
        if name.startswith("linear"):
            w, b = train_linear(A, ytr, float(name.split()[-1]))
            prob = lambda M: sigmoid(M @ w + b)
        else:
            P = train(A, ytr, n_hidden=16, lam=lam, seed=seed)
            prob = lambda M: sigmoid(forward(P, M)[2])
        for rows, M, t in [(val, B, yva), (test, C, yte), (train_loss, A, ytr)]:
            p = prob(M)
            rows[name].append((log_loss(p, t), ((p >= .5) == t).mean()))

corrected = np.sqrt(1 / 20 + 40 / 120)            # Nadeau-Bengio, "Features and Overfitting"
base = np.array(val["linear, L2 0.003"])
print(f"{'':20s} {'train':>6s} {'val loss':>9s} {'vs linear 0.003':>16s} {'val acc':>8s}")
for name in models:
    r, t = np.array(val[name]), np.array(train_loss[name])
    d = r[:, 0] - base[:, 0]
    vs = f"{d.mean():+.3f} ± {d.std(ddof=1) * corrected:.3f}" if d.any() else ""
    print(f"{name:20s} {t[:, 0].mean():6.3f} {r[:, 0].mean():9.3f} {vs:>16s} {r[:, 1].mean():8.3f}")

best = min((m for m in models if m.startswith("16")), key=lambda m: np.mean(np.array(val[m])[:, 0]))
a, b = np.array(test[best])[:, 0], np.array(test["linear, L2 0.003"])[:, 0]
print(f"\nchosen on validation: {best}. test loss {a.mean():.3f} vs linear {b.mean():.3f}, "
      f"difference {(a - b).mean():+.3f} ± {(a - b).std(ddof=1) * corrected:.3f}")