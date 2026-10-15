# decay.py
import sys
import numpy as np
from router_data import split
from router4 import four_way
from softmax_basics import softmax, cross_entropy
from optim import Adam, SGD

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000   # steps per run

class L2SGD(SGD):
    """Features and Overfitting's L2 penalty with plain mini-batch gradient descent."""
    def __init__(self, eta, lam):
        super().__init__(eta)
        self.lam = lam
    def step(self, p, g, key):
        super().step(p, g + (self.lam * p if key == "W" else 0), key)

def train(opt, X, y, seed, steps=STEPS, batch=16):
    rng = np.random.default_rng(seed)
    W, b = np.zeros((X.shape[1], 4)), np.zeros(4)
    for _ in range(steps):
        i = rng.choice(len(y), batch, replace=False)
        G = (softmax(X[i] @ W + b) - np.eye(4)[y[i]]) / batch
        opt.step(W, X[i].T @ G, "W")                    # the optimizer adds any decay itself
        opt.step(b, G.sum(0), "b")
    return W, b

METHODS = [("no decay", lambda lam: Adam(0.01), [0]),
           ("SGD + L2", lambda lam: L2SGD(0.5, lam), [0.0003, 0.001, 0.003]),
           ("Adam + L2", lambda lam: Adam(0.01, decay=lam, decoupled=False), [0.0001, 0.0003, 0.001, 0.003]),
           ("AdamW", lambda lam: Adam(0.01, decay=lam), [0.1, 0.3, 1])]
res = {}
for seed in range(20):
    ((Xtr, ytr), (Xva, yva), (Xte, yte)), _ = four_way(*split(seed))
    common = Xtr.sum(0) >= 5                                # words in 5+ training questions
    for name, make, lams in METHODS:
        for lam in lams:
            W, b = train(make(lam), Xtr, ytr, seed)
            size = lambda mask: np.sqrt((W[mask] ** 2).mean())
            res.setdefault((name, lam), []).append(
                (cross_entropy(Xva @ W + b, yva), size(common), size(~common),
                 cross_entropy(Xte @ W + b, yte), ((Xte @ W + b).argmax(1) == yte).mean()))
print(f"{STEPS:,} steps, mini-batches of 16")
for (name, lam), rows in res.items():
    r = np.array(rows)
    print(f"{name:9s} λ {lam:<6} val CE {r[:, 0].mean():.3f}   weight size: common words "
          f"{r[:, 1].mean():.3f}, rare words {r[:, 2].mean():.3f}, ratio {np.mean(r[:, 1] / r[:, 2]):.2f}")

corrected = np.sqrt(1 / 20 + 40 / 120)
se = lambda d: f"{d.mean():+.3f} ± {d.std(ddof=1) * corrected:.3f}"
best = {n: min((k for k in res if k[0] == n), key=lambda k: np.mean(np.array(res[k])[:, 0]))
        for n, _, _ in METHODS}
test = {n: np.array(res[k]) for n, k in best.items()}
print("chosen on validation: " + ", ".join(f"{n} λ {k[1]}" for n, k in best.items() if n != "no decay"))
print("on test, at the chosen λ:  " + ", ".join(f"{n} CE {t[:, 3].mean():.3f} acc {t[:, 4].mean():.3f}"
                                                for n, t in test.items()))
for a, b in [("AdamW", "SGD + L2"), ("AdamW", "no decay"), ("SGD + L2", "no decay")]:
    print(f"test, {a} minus {b}: CE {se(test[a][:, 3] - test[b][:, 3])}, "
          f"accuracy {se(test[a][:, 4] - test[b][:, 4])}")
