import numpy as np
from router_data import split, featurize
from overfit import sigmoid, log_loss, train

# 6. How sure are we? Compare two settings on the same 20 splits (paired),
#    with the Nadeau-Bengio correction for splits that share questions.
def val_loss(lam, steps=4000):
    out = []
    for seed in range(20):
        ((Xtr, ytr), (Xva, yva), _), _ = featurize(*split(seed))
        w, b = train(Xtr, ytr, lam, steps=steps)
        out.append(log_loss(sigmoid(Xva @ w + b), yva))
    return np.array(out)

def compare(a, b, label):
    d = a - b
    naive = d.std(ddof=1) / np.sqrt(len(d))
    honest = d.std(ddof=1) * np.sqrt(1 / len(d) + 40 / 120)   # n_val / n_train
    print(f"{label:37s} {d.mean():+.3f}   naive SE {naive:.3f}   corrected SE {honest:.3f}")

l2 = val_loss(0.003)
compare(l2, val_loss(0.0), "L2 0.003 - no penalty (4,000 steps)")
compare(l2, val_loss(0.0, steps=16000), "L2 0.003 - no penalty (16,000 steps)")
compare(l2, val_loss(0.001), "L2 0.003 - L2 0.001")
compare(l2, val_loss(0.01), "L2 0.003 - L2 0.01")