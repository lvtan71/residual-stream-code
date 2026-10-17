import numpy as np

EPS = 1e-5

def layernorm(x, g, b):
    """Normalize each row to mean 0, variance 1, then scale by g and shift by b."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + EPS)
    return xhat * g + b, (xhat, var)

def layernorm_back(dy, g, cache):
    xhat, var = cache
    dxhat = dy * g
    # The gradient of a normalization removes the parts of dxhat that would
    # only move the mean or the overall scale, which the normalization undoes.
    dx = (dxhat - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / np.sqrt(var + EPS)
    return dx, (dy * xhat).sum(0), dy.sum(0)            # dx, dg, db

def rmsnorm(x, g):
    """Divide each row by its root mean square, then scale by g. No mean, no shift."""
    rms = np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS)
    xhat = x / rms
    return xhat * g, (xhat, rms)

def rmsnorm_back(dy, g, cache):
    xhat, rms = cache
    dxhat = dy * g
    dx = (dxhat - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / rms
    return dx, (dy * xhat).sum(0)                        # dx, dg

def batchnorm(x, g, b):
    """Normalize each *column* over the batch: every example depends on the others."""
    mu, var = x.mean(0), x.var(0)
    xhat = (x - mu) / np.sqrt(var + EPS)
    return xhat * g + b, (mu, var)

if __name__ == "__main__":
    # A worked vector, then the same vector shifted by 100.
    g, b = np.ones(4), np.zeros(4)
    for x in (np.array([[2.0, 4.0, 6.0, 8.0]]), np.array([[102.0, 104.0, 106.0, 108.0]])):
        print(f"x = {x[0]}:  LayerNorm {np.round(layernorm(x, g, b)[0][0], 3)}   "
              f"RMSNorm {np.round(rmsnorm(x, g)[0][0], 3)}")
    # Check both backward passes against finite differences, as in "Backpropagation, by Hand".
    rng = np.random.default_rng(0)
    x, g, b, dy = rng.normal(size=(3, 5)), rng.normal(size=5), rng.normal(size=5), rng.normal(size=(3, 5))
    for name, f, back in [("LayerNorm", lambda x: layernorm(x, g, b), lambda c: layernorm_back(dy, g, c)[0]),
                          ("RMSNorm", lambda x: rmsnorm(x, g), lambda c: rmsnorm_back(dy, g, c)[0])]:
        analytic, worst = back(f(x)[1]), 0.0
        for i in np.ndindex(x.shape):
            e = np.zeros_like(x); e[i] = 1e-6
            numeric = ((f(x + e)[0] - f(x - e)[0]) * dy).sum() / 2e-6
            worst = max(worst, abs(numeric - analytic[i]) / max(1e-12, abs(numeric) + abs(analytic[i])))
        print(f"{name} backward vs finite differences: largest relative error {worst:.1e}")
