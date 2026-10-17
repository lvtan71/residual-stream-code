import numpy as np
from router_data import split, featurize
from overfit import sigmoid, log_loss, train

# 2. Average every setting over 20 random splits: one 40-question
#    validation set is too small to trust on its own.
def evaluate(lam, penalty, seeds=range(20)):
    tr_acc, va_loss, va_acc, nonzero = [], [], [], []
    for seed in seeds:
        ((Xtr, ytr), (Xva, yva), _), _ = featurize(*split(seed))
        w, b = train(Xtr, ytr, lam, penalty)
        p = sigmoid(Xva @ w + b)
        tr_acc.append(((sigmoid(Xtr @ w + b) >= 0.5) == ytr).mean())
        va_loss.append(log_loss(p, yva))
        va_acc.append(((p >= 0.5) == yva).mean())
        nonzero.append((np.abs(w) > 0).sum())
    se = np.std(va_loss) / np.sqrt(len(va_loss))
    return (f"{penalty} λ={lam:<6}: train acc {np.mean(tr_acc):.3f}  "
            f"val loss {np.mean(va_loss):.3f} ± {se:.3f}  "
            f"val acc {np.mean(va_acc):.3f}  nonzero weights {np.mean(nonzero):.0f}")

for lam in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]:
    print(evaluate(lam, "l2"))
for lam in [0.001, 0.004, 0.008, 0.016]:
    print(evaluate(lam, "l1"))