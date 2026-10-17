import numpy as np
from collections import Counter
from mlp import sigmoid, forward, backward, init, train

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)

# Why do 82 of 100 two-unit starts fail? Classify every failure.
outcome, dead_start, solved_if_dead = Counter(), 0, 0
for seed in range(100):
    W1 = init(2, 2, np.random.default_rng(seed))["W1"]
    dead = ((W1 <= 0).all(axis=0)).any()          # a unit with both weights <= 0 never fires
    dead_start += dead
    P = train(X, y, 2, steps=5000, eta=1.0, seed=seed)
    right = int(((sigmoid(forward(P, X)[2]) >= 0.5) == y).sum())
    solved_if_dead += dead and right == 4
    if right < 4:
        grads, loss = backward(P, X, y)
        size = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        outcome[(right, round(loss, 3), "dead unit at start" if dead else "both units alive",
                 "gradient < 1e-3" if size < 1e-3 else "gradient >= 1e-3")] += 1
print(f"starts with a unit dead from step 0: {dead_start}/100, of which solved: {solved_if_dead}")
for (right, loss, dead, grad), n in sorted(outcome.items(), key=lambda kv: -kv[1]):
    print(f"{n:3d} failures: {right}/4 right, loss {loss:.3f}, {dead}, {grad}")