import numpy as np
from router_data import split, featurize
from overfit import sigmoid, log_loss, train

# 5. Choose λ by the test set instead of the validation set, using the SAME
#    criterion (log-loss) on both sides, and report test log-loss.
LAMBDAS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
peek = []
for seed in range(20):
    ((Xtr, ytr), (Xva, yva), (Xte, yte)), _ = featurize(*split(seed))
    fits = [train(Xtr, ytr, lam) for lam in LAMBDAS]
    test_loss = [log_loss(sigmoid(Xte @ w + b), yte) for w, b in fits]
    val_loss = [log_loss(sigmoid(Xva @ w + b), yva) for w, b in fits]
    peek.append((min(test_loss), test_loss[int(np.argmin(val_loss))]))
peek = np.array(peek)
print(f"test loss, λ chosen on test: {peek[:, 0].mean():.3f};"
      f" chosen on validation: {peek[:, 1].mean():.3f}")
