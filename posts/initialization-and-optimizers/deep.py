# deep.py
import numpy as np
from softmax_basics import softmax, cross_entropy

def init(sizes, rng):
    """He initialization for every ReLU layer: std sqrt(2 / fan_in)."""
    return [{"W": rng.normal(0, np.sqrt(2 / m), (m, n)), "b": np.zeros(n)}
            for m, n in zip(sizes[:-1], sizes[1:])]

def loss_and_grads(layers, X, y):
    """Softmax cross-entropy of a ReLU network, and its gradient ("Backpropagation, by Hand"'s rules)."""
    hs, h = [X], X
    for i, L in enumerate(layers):
        z = h @ L["W"] + L["b"]
        h = z if i == len(layers) - 1 else np.maximum(z, 0)
        hs.append(h)
    g = (softmax(hs[-1]) - np.eye(hs[-1].shape[1])[y]) / len(y)      # p - y, "Cross-Entropy and the Softmax Gradient"
    grads = [None] * len(layers)
    for i in reversed(range(len(layers))):
        grads[i] = {"W": hs[i].T @ g, "b": g.sum(0)}
        g = (g @ layers[i]["W"].T) * (hs[i] > 0)                         # back through ReLU
    return cross_entropy(hs[-1], y), grads
