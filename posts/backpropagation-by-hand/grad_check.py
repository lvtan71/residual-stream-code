import numpy as np
from mlp import init, backward

# Compare backprop to finite differences on a small random problem.
rng = np.random.default_rng(1)
X, y = rng.normal(size=(5, 4)), rng.integers(0, 2, 5).astype(float)
P = init(4, 3, rng)
grads, _ = backward(P, X, y, lam=0.01)
eps, worst = 1e-6, 0.0
for k in ["W1", "b1", "w2"]:
    for i in np.ndindex(P[k].shape):
        old = P[k][i]
        P[k][i] = old + eps; up = backward(P, X, y, 0.01)[1]
        P[k][i] = old - eps; down = backward(P, X, y, 0.01)[1]
        P[k][i] = old
        numeric = (up - down) / (2 * eps)                 # central difference
        worst = max(worst, abs(numeric - grads[k][i]) / max(1e-12, abs(numeric) + abs(grads[k][i])))
print(f"largest relative error, backprop vs finite differences: {worst:.1e}")