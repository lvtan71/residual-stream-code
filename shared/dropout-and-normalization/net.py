import numpy as np
from softmax_basics import softmax, cross_entropy
from norms import layernorm, layernorm_back, rmsnorm, rmsnorm_back

def init(sizes, rng, std=lambda m: np.sqrt(2 / m), norm=None):
    layers = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        L = {"W": rng.normal(0, std(m), (m, n)), "b": np.zeros(n)}
        if norm and len(layers) < len(sizes) - 2:           # hidden layers only
            L["g"] = np.ones(n)
            if norm == "layer":
                L["beta"] = np.zeros(n)
        layers.append(L)
    return layers

def forward(layers, X, norm=None, drop=0.0, rng=None):
    """Each hidden layer: linear, then (optionally) a norm, ReLU, then dropout."""
    h, caches = X, []
    for i, L in enumerate(layers):
        z = h @ L["W"] + L["b"]
        if i == len(layers) - 1:
            caches.append((h, None, None, None))
            return z, caches
        nc = None
        if norm == "layer":
            z, nc = layernorm(z, L["g"], L["beta"])
        elif norm == "rms":
            z, nc = rmsnorm(z, L["g"])
        a = np.maximum(z, 0)
        mask = None
        if drop and rng is not None:                       # inverted dropout, training only
            mask = (rng.random(a.shape) >= drop) / (1 - drop)
            a = a * mask
        caches.append((h, z, nc, mask))
        h = a

def loss_and_grads(layers, X, y, norm=None, drop=0.0, rng=None):
    logits, caches = forward(layers, X, norm, drop, rng)
    g = (softmax(logits) - np.eye(logits.shape[1])[y]) / len(y)      # p - y
    grads = [None] * len(layers)
    for i in reversed(range(len(layers))):
        h, z, nc, mask = caches[i]
        if i < len(layers) - 1:                            # back through dropout, ReLU, norm
            if mask is not None:
                g = g * mask
            g = g * (z > 0)
            G = {}
            if norm == "layer":
                g, G["g"], G["beta"] = layernorm_back(g, layers[i]["g"], nc)
            elif norm == "rms":
                g, G["g"] = rmsnorm_back(g, layers[i]["g"], nc)
        else:
            G = {}
        G.update({"W": h.T @ g, "b": g.sum(0)})
        grads[i] = G
        g = g @ layers[i]["W"].T
    return cross_entropy(logits, y), grads

def fit(layers, opt, X, y, steps, rng, norm=None, drop=0.0, batch=16, every=None):
    """Mini-batch training. Returns the full training loss at the steps in `every`."""
    seen = {}
    for t in range(1, steps + 1):
        idx = rng.choice(len(y), batch, replace=False)
        _, grads = loss_and_grads(layers, X[idx], y[idx], norm, drop, rng)
        for i, (L, G) in enumerate(zip(layers, grads)):
            for k in L:
                opt.step(L[k], G[k], (i, k))
        if every and t in every:
            seen[t] = cross_entropy(forward(layers, X, norm)[0], y)
    return seen
