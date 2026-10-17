import numpy as np
from a_loss import words as x, tokens as y

def gradient_descent(x, y, eta, steps):
    w, b = 0.0, 0.0
    for t in range(steps):
        err = w * x + b - y               # prediction minus target
        w -= eta * 2 * (err * x).mean()   # dL/dw
        b -= eta * 2 * err.mean()         # dL/db
    return w, b, ((w * x + b - y) ** 2).mean()

for eta in [0.01, 0.05, 0.07]:
    w, b, mse = gradient_descent(x, y, eta, steps=400)
    print(f"raw words, eta={eta:<5}: w={w:9.4g}  b={b:9.4g}  MSE={mse:.4g}")

xs = (x - x.mean()) / x.std()             # standardize the feature
for eta in [0.5, 1.0, 1.05]:
    w, b, mse = gradient_descent(xs, y, eta, steps=400)
    print(f"standardized, eta={eta:<4}: w={w:9.4g}  b={b:9.4g}  MSE={mse:.4g}")