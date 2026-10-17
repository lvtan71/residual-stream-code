import numpy as np

# Bradley-Terry on three answers, A, B and C. People compared them in pairs:
# A beat B in 3 of 4 comparisons, B beat C in 3 of 4, A beat C in 4 of 5.
# Model: P(i beats j) = sigmoid(r_i - r_j). Fit r by maximizing the log-
# likelihood of the 13 outcomes. Only differences matter, so fix r_C = 0.
wins = {("A", "B"): (3, 1), ("B", "C"): (3, 1), ("A", "C"): (4, 1)}    # (i wins, j wins)
sigmoid = lambda z: 1 / (1 + np.exp(-z))
r = {"A": 0.0, "B": 0.0, "C": 0.0}
for _ in range(5000):
    grad = {k: 0.0 for k in r}
    for (i, j), (wi, wj) in wins.items():
        p = sigmoid(r[i] - r[j])
        grad[i] += wi * (1 - p) - wj * p                  # d/dr_i of wi log p + wj log(1 - p)
        grad[j] -= wi * (1 - p) - wj * p
    for k in ("A", "B"):
        r[k] += 0.05 * grad[k]
print("rewards: " + ", ".join(f"r_{k} = {v:+.2f}" for k, v in r.items()))
for (i, j), (wi, wj) in wins.items():
    print(f"P({i} beats {j}) = sigmoid({r[i]:.2f} - {r[j]:.2f}) = {sigmoid(r[i] - r[j]):.2f}   observed {wi}/{wi + wj} = {wi / (wi + wj):.2f}")
