import numpy as np
from net import init, loss_and_grads

# The whole network's backward pass, with each norm, against finite differences.
rng = np.random.default_rng(0)
X, y = rng.normal(size=(6, 5)), rng.integers(0, 4, 6)
for norm in (None, "layer", "rms"):
    layers, worst = init([5, 7, 7, 4], rng, norm=norm), 0.0
    _, grads = loss_and_grads(layers, X, y, norm)
    for L, G in zip(layers, grads):
        for k in L:
            for i in np.ndindex(L[k].shape):
                old = L[k][i]
                L[k][i] = old + 1e-6; up = loss_and_grads(layers, X, y, norm)[0]
                L[k][i] = old - 1e-6; down = loss_and_grads(layers, X, y, norm)[0]
                L[k][i] = old
                num = (up - down) / 2e-6
                worst = max(worst, abs(num - G[k][i]) / max(1e-12, abs(num) + abs(G[k][i])))
    print(f"norm={norm}: largest relative error {worst:.1e}")
