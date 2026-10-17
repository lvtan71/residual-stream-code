# race.py
import sys
import numpy as np
from router_data import split
from router4 import four_way
from softmax_basics import softmax, cross_entropy
from optim import SGD, Momentum, Adam

# The four-way router of "Cross-Entropy and the Softmax Gradient" with an L2 penalty: a convex objective, so
# "better optimizer" can only mean "lower objective sooner".
LAM = 0.003
def objective(W, b, X, y):
    return cross_entropy(X @ W + b, y) + 0.5 * LAM * (W ** 2).sum()

def run(opt, X, y, checkpoints=(30, 300, 3000)):
    W, b, out = np.zeros((X.shape[1], 4)), np.zeros(4), []
    for t in range(1, max(checkpoints) + 1):
        G = (softmax(X @ W + b) - np.eye(4)[y]) / len(y)
        opt.step(W, X.T @ G + LAM * W, "W")
        opt.step(b, G.sum(0), "b")
        if t in checkpoints:
            out.append(objective(W, b, X, y))
    return out

if __name__ == "__main__":
    with_length = len(sys.argv) > 1                       # add one raw feature: question length in characters
    print("plus raw question length:" if with_length else "bag of words:")
    etas = {"gradient descent": (SGD, [10 ** k for k in np.arange(-4, 1.6, 0.5)]),
            "momentum 0.9": (Momentum, [10 ** k for k in np.arange(-5, 1.1, 0.5)]),
            "Adam": (Adam, [10 ** k for k in np.arange(-3, 0.6, 0.5)])}
    for name, (make, grid) in etas.items():
        best = {}
        for eta in grid:
            rows = []
            for seed in range(5):
                tr, va, te = split(seed)
                ((X, y), _, _), _ = four_way(tr, va, te)
                if with_length:
                    X = np.c_[X, [len(r["question"]) for r in tr]]
                with np.errstate(over="ignore", invalid="ignore"):
                    rows.append(run(make(eta), X, y))
            r = np.nan_to_num(np.median(rows, axis=0), nan=np.inf)
            for k, v in enumerate(r):
                if v < best.get(k, (np.inf,))[0]:
                    best[k] = (v, eta)
        print(f"{name:17s} " + "   ".join(f"after {n:>4} steps {best[k][0]:.4f} (eta {best[k][1]:.2g})"
                                           for k, n in enumerate((30, 300, 3000))))
