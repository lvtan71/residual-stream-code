import numpy as np
from router_data import split
from router4 import four_way
from net import init, fit
from optim import SGD, Adam

# The ten-layer router of "Initialization and Optimizers", from its fatal start (std 0.01) and from He,
# with and without a normalization layer in every hidden layer. For each,
# sweep the learning rate and keep the best: can *any* step size train it?
grids = {"SGD": (SGD, [1e-4, 1e-3, 1e-2, 1e-1, 1.0]), "Adam": (Adam, [1e-5, 1e-4, 1e-3, 1e-2])}
out = {}
for seed in range(5):
    ((X, y), _, _), _ = four_way(*split(seed))
    sizes = [X.shape[1]] + [64] * 10 + [4]
    for start, std in [("std 0.01", lambda m: 0.01), ("He", lambda m: np.sqrt(2 / m))]:
        for norm in (None, "layer", "rms"):
            for opt_name, (make, etas) in grids.items():
                for eta in etas:
                    rng = np.random.default_rng(seed)
                    layers = init(sizes, rng, std, norm)
                    with np.errstate(all="ignore"):
                        seen = fit(layers, make(eta), X, y, 1000, rng, norm, every={1000})
                    out.setdefault((start, norm or "none", opt_name, eta), []).append(seen[1000])
print(f"{'start':9s} {'norm':6s} {'optimizer':5s}  train loss after 1,000 steps, by learning rate")
for (start, norm, opt_name), (make, etas) in [((s, n, o), grids[o]) for s in ("std 0.01", "He")
                                              for n in ("none", "layer", "rms") for o in grids]:
    cells = [f"{e:g}: {np.nan_to_num(np.median(out[(start, norm, opt_name, e)]), nan=np.inf):.3f}" for e in etas]
    print(f"{start:9s} {norm:6s} {opt_name:5s}  " + "  ".join(cells))
