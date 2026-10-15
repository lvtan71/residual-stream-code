import numpy as np
from router_data import split, featurize

sigmoid = lambda z: 1 / (1 + np.exp(-z))

def log_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

def train(X, y, lam=0.0, penalty="l2", eta=0.5, steps=4000):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        grad_w, grad_b = X.T @ (p - y) / len(y), (p - y).mean()
        if penalty == "l2":                   # weight decay
            w -= eta * (grad_w + lam * w)
        else:                                 # L1: step, then soft-threshold
            w -= eta * grad_w
            w = np.sign(w) * np.maximum(np.abs(w) - eta * lam, 0.0)
        b -= eta * grad_b
    return w, b

if __name__ == "__main__":
    # 1. One split, no penalty: watch training and validation loss diverge.
    (train_set, val_set, _), vocab = featurize(*split(seed=0))
    (Xtr, ytr), (Xva, yva) = train_set, val_set
    print(f"{len(vocab)} words, {len(ytr)} training questions")
    w, b = np.zeros(Xtr.shape[1]), 0.0
    for step in range(1, 25601):
        p = sigmoid(Xtr @ w + b)
        w -= 0.5 * Xtr.T @ (p - ytr) / len(ytr)
        b -= 0.5 * (p - ytr).mean()
        if step in (100, 400, 1600, 3200, 6400, 25600):
            print(f"step {step:5d}: train loss {log_loss(sigmoid(Xtr @ w + b), ytr):.4f}"
                  f"  val loss {log_loss(sigmoid(Xva @ w + b), yva):.4f}"
                  f"  |w| {np.linalg.norm(w):5.2f}")
