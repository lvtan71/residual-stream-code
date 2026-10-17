import numpy as np
from router_data import split
from softmax_basics import softmax
from router4 import four_way, train

def one_vs_rest(X, y, lam=0.003, steps=4000, eta=0.5):
    """Four independent sigmoid classifiers, one per route."""
    W, b = np.zeros((X.shape[1], 4)), np.zeros(4)
    Y = np.eye(4)[y]
    for _ in range(steps):
        P = 1 / (1 + np.exp(-(X @ W + b)))
        G = (P - Y) / len(y)                            # the same p - y, per column
        W -= eta * (X.T @ G + lam * W)
        b -= eta * G.sum(axis=0)
    return W, b

ce = lambda P, y: -np.log(P[np.arange(len(y)), y]).mean()
rows = []
for seed in range(20):
    ((Xtr, ytr), (Xva, yva), _), _ = four_way(*split(seed))
    Ws, bs = train(Xtr, ytr, lam=0.003)
    Wo, bo = one_vs_rest(Xtr, ytr)
    P = softmax(Xva @ Ws + bs)
    S = 1 / (1 + np.exp(-(Xva @ Wo + bo)))
    rows.append(((P.argmax(1) == yva).mean(), (S.argmax(1) == yva).mean(),
                 np.abs(S.sum(1) - 1).mean(), (S.max(1) < 0.5).mean(), (P.max(1) < 0.5).mean(),
                 ce(P, yva), ce(S / S.sum(1, keepdims=True), yva)))
r = np.array(rows)
se = lambda d: f"{d.mean():+.3f} ± {d.std(ddof=1) * np.sqrt(1 / 20 + 40 / 120):.3f}"
print(f"val accuracy: softmax {r[:, 0].mean():.3f}, one-vs-rest {r[:, 1].mean():.3f} "
      f"(difference {se(r[:, 1] - r[:, 0])})")
print(f"one-vs-rest scores: sum is off from 1 by {r[:, 2].mean():.2f} on average; no route above 0.5 "
      f"on {r[:, 3].mean():.0%} of questions (softmax: {r[:, 4].mean():.0%})")
print(f"val CE: softmax {r[:, 5].mean():.3f}, one-vs-rest rescaled to sum to 1 {r[:, 6].mean():.3f} "
      f"(difference {se(r[:, 6] - r[:, 5])})")