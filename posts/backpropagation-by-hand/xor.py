import numpy as np
from mlp import sigmoid, relu, forward, backward, train

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)          # XOR

# One example, one step, every number.
P = {"W1": np.array([[0.5, -0.4], [0.3, 0.8]]), "b1": np.array([0.0, 0.1]),
     "w2": np.array([0.7, -0.6]), "b2": 0.1}
x, t = X[2:3], y[2:3]                             # x = (1, 0), target 1
z1, a1, z2 = forward(P, x)
grads, loss = backward(P, x, t)
print(f"z1 {z1[0]}  a1 {a1[0]}  z2 {z2[0]:.4f}  p {sigmoid(z2[0]):.4f}  loss {loss:.4f}")
for k in ["w2", "b2", "W1", "b1"]:
    print(f"dL/d{k} = {np.round(grads[k], 4).tolist()}")
P = {k: v - 0.5 * grads[k] for k, v in P.items()}
print(f"after one step: p {sigmoid(forward(P, x)[2][0]):.4f}  loss {backward(P, x, t)[1]:.4f}")

# How often does training find XOR, by hidden-layer width?
for h in [2, 3, 4, 8, 16]:
    solved = 0
    for seed in range(100):
        P = train(X, y, h, steps=5000, eta=1.0, seed=seed)
        solved += ((sigmoid(forward(P, X)[2]) >= 0.5) == y).all()
    print(f"{h:2d} hidden units: XOR solved from {solved}/100 random starts")