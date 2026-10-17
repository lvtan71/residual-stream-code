import numpy as np
from router_data import split
from router4 import four_way

# The largest curvature of race.py's objective at the start (W = 0), where the
# softmax Hessian block is (diag(p) - p p^T) with p uniform, largest eigenvalue 1/4.
tr, va, te = split(0)
((X, y), _, _), _ = four_way(tr, va, te)
for name, Z in [("bag of words", X), ("plus raw length", np.c_[X, [len(r["question"]) for r in tr]])]:
    Zb = np.c_[Z, np.ones(len(Z))]
    top = 0.25 * np.linalg.eigvalsh(Zb.T @ Zb / len(Zb))[-1] + 0.003
    print(f"{name:16s} largest curvature {top:8.2f}, so gradient descent is stable only below 2 / that = {2 / top:.4f}")
