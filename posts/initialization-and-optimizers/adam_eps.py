import numpy as np
from router_data import split
from router4 import four_way
from deep import loss_and_grads
from deep_init import init
from optim import Adam

# Is Adam's epsilon what keeps it stuck at the tiny start? Remove it and see.
sizes = [None] + [64] * 10 + [4]
for eps in (1e-8, 1e-30):
    finals = []
    for seed in range(5):
        ((X, y), _, _), _ = four_way(*split(seed))
        sizes[0] = X.shape[1]
        rng = np.random.default_rng(seed)
        layers, opt = init(sizes, rng, lambda m: 0.01), Adam(1e-3, eps=eps)
        for t in range(1000):
            idx = rng.choice(len(y), 16, replace=False)
            _, grads = loss_and_grads(layers, X[idx], y[idx])
            for i, (L, G) in enumerate(zip(layers, grads)):
                for k in L:
                    opt.step(L[k], G[k], (i, k))
        finals.append(loss_and_grads(layers, X, y)[0])
    print(f"Adam eps {eps:g}: train loss after 1,000 steps, 5 seeds: " + " ".join(f"{v:.3f}" for v in finals))
