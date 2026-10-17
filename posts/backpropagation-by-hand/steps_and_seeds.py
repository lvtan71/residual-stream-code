import numpy as np
from mlp import sigmoid, forward, train
from overfit import log_loss
from router_data import split, featurize
from router_models import keyword_count

# Does the answer depend on the step count or on the initialization seed?
runs = {}
for seed in range(20):
    tr, va, te = split(seed)
    ((Xtr, ytr), (Xva, yva), _), _ = featurize(tr, va, te)
    A, B = np.c_[Xtr, keyword_count(tr)], np.c_[Xva, keyword_count(va)]
    for lam in [0.0, 0.03]:
        for steps in [3000, 12000]:
            for init_seed in range(5):
                P = train(A, ytr, n_hidden=16, lam=lam, steps=steps, seed=100 * init_seed + seed)
                runs.setdefault((lam, steps, init_seed), []).append(
                    log_loss(sigmoid(forward(P, B)[2]), yva))
for lam in [0.0, 0.03]:
    for steps in [3000, 12000]:
        means = [np.mean(runs[(lam, steps, s)]) for s in range(5)]
        print(f"L2 {lam:<4} {steps:6,d} steps: val loss {min(means):.3f}-{max(means):.3f} "
              f"across 5 initialization seeds")