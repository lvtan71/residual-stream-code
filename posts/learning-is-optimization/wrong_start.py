import numpy as np
from c_router import x, y, sigmoid

# Start confidently wrong (w = -5) and train with each loss, same steps.
for name in ["log-loss", "squared error"]:
    w, b = -5.0, 0.0
    for step in range(500):
        p = sigmoid(w * x + b)
        grad_z = (p - y) if name == "log-loss" else 2 * (p - y) * p * (1 - p)
        w -= 0.5 * (grad_z * x).mean()
        b -= 0.5 * grad_z.mean()
    p = sigmoid(w * x + b)
    print(f"{name:13s} after 500 steps: w = {w:6.3f}, b = {b:6.3f},"
          f" accuracy = {((p >= 0.5) == y).mean():.3f}")