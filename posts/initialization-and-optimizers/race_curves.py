import numpy as np
from router_data import split
from router4 import four_way
from optim import SGD, Momentum, Adam
from race import run

# The curves behind the chart: bag of words plus raw question length, each
# optimizer at its best learning rate for 3,000 steps (from race.py), plus
# momentum at gradient descent's learning rate.
marks = [int(round(10 ** k)) for k in np.arange(0, 3.51, 0.25)]
for name, make in [("gradient descent", lambda: SGD(10 ** -2.5)),
                   ("momentum 0.9, same eta", lambda: Momentum(10 ** -2.5)),
                   ("momentum 0.9, eta 0.01", lambda: Momentum(0.01)), ("Adam", lambda: Adam(0.01))]:
    rows = []
    for seed in range(5):
        tr, va, te = split(seed)
        ((X, y), _, _), _ = four_way(tr, va, te)
        X = np.c_[X, [len(r["question"]) for r in tr]]
        rows.append(run(make(), X, y, checkpoints=marks))
    print(f"{name}: " + " ".join(f"{v:.4f}" for v in np.median(rows, axis=0)))
print("steps: " + " ".join(map(str, marks)))
