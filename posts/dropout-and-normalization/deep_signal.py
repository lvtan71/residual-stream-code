import numpy as np
from norms import layernorm, layernorm_back

def signal(std, depth=20, width=256, seed=0):
    """The 20-layer test from "Initialization and Optimizers", with a LayerNorm before every ReLU."""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(512, width))
    Ws = [rng.normal(0, std, (width, width)) for _ in range(depth)]
    g, b = np.ones(width), np.zeros(width)
    caches, hs = [], []
    for W in Ws:
        hs.append(h)
        z, c = layernorm(h @ W, g, b)
        caches.append((z, c))
        h = np.maximum(z, 0)
    top = np.sqrt((h ** 2).mean())
    d = rng.normal(size=h.shape)
    wgrad = []
    for W, (z, c), hin in zip(Ws[::-1], caches[::-1], hs[::-1]):
        d = layernorm_back(d * (z > 0), g, c)[0]
        wgrad.append(np.sqrt(((hin.T @ d) ** 2).mean()))
        d = d @ W.T
    return top, np.sqrt((d ** 2).mean()), wgrad[-1]

print(f"{'weight std':>10s}  {'activation size, layer 20':>26s}  {'gradient reaching layer 1':>26s}  "
      f"{'gradient on W, layer 1':>23s}")
for std in (0.01, np.sqrt(2 / 256), 1.0):
    a, gx, gw = np.median([signal(std, seed=s) for s in range(5)], axis=0)
    print(f"{std:10.3f}  {a:26.2f}  {gx:26.2e}  {gw:23.2e}")
