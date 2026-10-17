import numpy as np

def signal(scale, act, depth=20, width=256, seed=0):
    """Push a batch of 512 random inputs through `depth` layers. Return the size
    of the activations after each layer, and of the gradient arriving at each
    layer on the way back, starting from a random unit-size gradient at the top."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(512, width))
    Ws = [rng.normal(0, scale(width), (width, width)) for _ in range(depth)]
    phi, dphi = {"relu": (lambda z: np.maximum(z, 0), lambda z: (z > 0) * 1.0),
                 "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2)}[act]
    rms = lambda a: np.sqrt((a ** 2).mean())
    h, zs, forward = x, [], []
    for W in Ws:
        zs.append(h @ W)
        h = phi(zs[-1])
        forward.append(rms(h))
    g, backward = rng.normal(size=h.shape), []
    for W, z in zip(Ws[::-1], zs[::-1]):
        g = (g * dphi(z)) @ W.T                        # "Backpropagation, by Hand": mask, then W transposed
        backward.append(rms(g))
    return forward, backward[::-1]                      # backward[0] reaches layer 1

schemes = {
    "N(0, 0.01^2), ReLU": (lambda n: 0.01, "relu"),
    "N(0, 1), ReLU":      (lambda n: 1.0, "relu"),
    "Xavier 1/n, ReLU":   (lambda n: np.sqrt(1 / n), "relu"),
    "He 2/n, ReLU":       (lambda n: np.sqrt(2 / n), "relu"),
    "Xavier 1/n, tanh":   (lambda n: np.sqrt(1 / n), "tanh"),
}
print(f"{'':19s} {'activation size after layer':^39s}  {'gradient reaching':^19s}")
print(f"{'':19s} {'1':>9s} {'5':>9s} {'10':>9s} {'20':>9s}  {'layer 20':>9s} {'layer 1':>9s}")
for name, (scale, act) in schemes.items():
    runs = [signal(scale, act, seed=s) for s in range(5)]
    f = np.median([r[0] for r in runs], axis=0)       # median over 5 seeds
    b = np.median([r[1] for r in runs], axis=0)
    print(f"{name:19s} " + " ".join(f"{f[k]:9.2e}" for k in (0, 4, 9, 19)) +
          f"  {b[19]:9.2e} {b[0]:9.2e}")
