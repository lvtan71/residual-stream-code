import numpy as np
from c_router import x, y, sigmoid

def log_loss(w, b):
    p = sigmoid(w * x + b)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

best = log_loss(1.746, -1.460)          # the full-batch optimum from above
for batch in [24, 4, 1]:
    finals, needed = [], []
    for seed in range(200):             # 200 different shuffles
        rng = np.random.default_rng(seed)
        w = b = 0.0
        updates, reached = 0, None
        while updates < 5000:
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch):
                i = order[start:start + batch]
                p = sigmoid(w * x[i] + b)
                w -= 0.5 * ((p - y[i]) * x[i]).mean()
                b -= 0.5 * (p - y[i]).mean()
                updates += 1
                if reached is None and log_loss(w, b) < 1.01 * best:
                    reached = updates
        finals.append(log_loss(w, b)); needed.append(reached)
    finals = np.array(finals)
    print(f"{batch:2d} per update: updates to reach 1% of best, median {np.median(needed):.0f};"
          f"  final loss median {np.median(finals):.4f}, worst {finals.max():.4f}")