import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))
relu = lambda z: np.maximum(z, 0.0)

def init(n_in, n_hidden, rng):
    """He initialization for the ReLU layer; small random output weights."""
    return {"W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)),
            "b1": np.zeros(n_hidden),
            "w2": rng.normal(0, np.sqrt(1 / n_hidden), n_hidden),
            "b2": 0.0}

def forward(P, X):
    z1 = X @ P["W1"] + P["b1"]            # (n, hidden)
    a1 = relu(z1)
    z2 = a1 @ P["w2"] + P["b2"]           # (n,)  one logit per example
    return z1, a1, z2

def backward(P, X, y, lam=0.0):
    """Gradients of the mean log-loss (+ L2 on weights) for every parameter."""
    n = len(y)
    z1, a1, z2 = forward(P, X)
    p = sigmoid(z2)
    dz2 = (p - y) / n                     # the p - y of post 1, per example
    grads = {"w2": a1.T @ dz2 + lam * P["w2"],
             "b2": dz2.sum()}
    da1 = np.outer(dz2, P["w2"])          # back through the output weights
    dz1 = da1 * (z1 > 0)                  # back through ReLU: pass or block
    grads["W1"] = X.T @ dz1 + lam * P["W1"]
    grads["b1"] = dz1.sum(axis=0)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    loss += 0.5 * lam * ((P["W1"] ** 2).sum() + (P["w2"] ** 2).sum())
    return grads, loss

def train(X, y, n_hidden, lam=0.0, steps=3000, eta=0.5, seed=0):
    P = init(X.shape[1], n_hidden, np.random.default_rng(seed))
    for _ in range(steps):
        grads, _ = backward(P, X, y, lam)
        for k in grads:
            P[k] = P[k] - eta * grads[k]
    return P
