import numpy as np
from router_data import split
from router4 import four_way
from deep import loss_and_grads
from optim import SGD, Adam

def init(sizes, rng, std):
    return [{"W": rng.normal(0, std(m), (m, n)), "b": np.zeros(n)}
            for m, n in zip(sizes[:-1], sizes[1:])]

if __name__ == "__main__":
    inits = {"N(0, 0.01^2)": lambda m: 0.01, "Xavier 1/n": lambda m: np.sqrt(1 / m),
             "He 2/n": lambda m: np.sqrt(2 / m)}
    sizes = [None] + [64] * 10 + [4]                         # ten hidden ReLU layers
    out = {}
    for seed in range(5):
        ((Xtr, ytr), _, _), _ = four_way(*split(seed))
        sizes[0] = Xtr.shape[1]
        for name, std in inits.items():
            for opt_name, make in [("SGD 0.1", lambda: SGD(0.1)), ("Adam 0.001", lambda: Adam(1e-3))]:
                rng = np.random.default_rng(seed)
                layers, opt = init(sizes, rng, std), make()
                start, g0 = loss_and_grads(layers, Xtr, ytr)
                first = g0[0]["W"]
                early = None
                for t in range(1000):
                    if t == 100:
                        early = loss_and_grads(layers, Xtr, ytr)[0]
                    idx = rng.choice(len(ytr), 16, replace=False)
                    _, grads = loss_and_grads(layers, Xtr[idx], ytr[idx])
                    for i, (L, G) in enumerate(zip(layers, grads)):
                        for k in L:
                            opt.step(L[k], G[k], (i, k))
                out.setdefault((name, opt_name), []).append(
                    (np.abs(first).max(), start, early, loss_and_grads(layers, Xtr, ytr)[0]))
    print(f"{'':14s} {'optimizer':11s} {'largest first-layer gradient':>29s}   train loss at 0 / 100 / 1,000 steps")
    for (name, opt_name), rows in out.items():
        r = np.median(rows, axis=0)
        print(f"{name:14s} {opt_name:11s} {r[0]:29.1e}   {r[1]:12.3f} {r[2]:7.3f} {r[3]:7.3f}")
