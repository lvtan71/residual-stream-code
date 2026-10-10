import numpy as np
from router_data import featurize, ROUTES
from softmax_basics import softmax

def four_way(train, *others):
    """Post 2.1's bag of words, with the route index (0-3) as the label."""
    sets, vocab = featurize(train, *others)
    return [(X, np.array([ROUTES.index(r["route"]) for r in rs]))
            for (X, _), rs in zip(sets, (train, *others))], vocab

def train(X, y, lam=0.0, smooth=0.0, steps=4000, eta=0.5, K=4):
    W, b = np.zeros((X.shape[1], K)), np.zeros(K)
    Y = np.eye(K)[y] * (1 - smooth) + smooth / K        # label smoothing
    for _ in range(steps):
        G = (softmax(X @ W + b) - Y) / len(y)           # p - y, per example
        W -= eta * (X.T @ G + lam * W)
        b -= eta * G.sum(axis=0)
    return W, b

def top_label_ece(P, y, bins=5):
    conf, right = P.max(axis=1), P.argmax(axis=1) == y
    which = np.clip((conf * bins).astype(int), 0, bins - 1)
    return sum((which == k).mean() * abs(conf[which == k].mean() - right[which == k].mean())
               for k in range(bins) if (which == k).any())
