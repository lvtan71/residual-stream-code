import numpy as np

def softmax(Z):
    """Row-wise softmax, shifted by the row max so exp never overflows."""
    Z = Z - Z.max(axis=-1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=-1, keepdims=True)

def cross_entropy(Z, y):
    """Mean of -log p[y], computed from logits with log-sum-exp."""
    Z = Z - Z.max(axis=-1, keepdims=True)
    log_p = Z - np.log(np.exp(Z).sum(axis=-1, keepdims=True))
    return -log_p[np.arange(len(y)), y].mean()

if __name__ == "__main__":
    # The gradient of cross-entropy w.r.t. the logits is p - onehot(y).
    rng = np.random.default_rng(0)
    z, target = rng.normal(size=(1, 4)), np.array([2])
    analytic = softmax(z)[0] - np.eye(4)[2]
    numeric = np.array([(cross_entropy(z + e, target) - cross_entropy(z - e, target)) / 2e-6
                        for e in np.eye(4)[:, None, :] * 1e-6])
    print(f"p - y:  {np.round(analytic, 4)}")
    print(f"largest gap to finite differences: {np.abs(numeric - analytic).max():.1e}")

    # Why the shift matters.
    z = np.array([[1000.0, 999.0, 0.0, -5.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(z) / np.exp(z).sum()
    print(f"naive softmax: {naive[0]}   shifted: {np.round(softmax(z)[0], 4)}")
    show = lambda v: "[" + " ".join(f"{x:.2f}" for x in v[0]) + "]"
    with np.errstate(divide="ignore"):
        print(f"log(softmax(z)): {show(np.log(softmax(z)))}   "
              f"log-sum-exp: {show(z - z.max() - np.log(np.exp(z - z.max()).sum()))}")