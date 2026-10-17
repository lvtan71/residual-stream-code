import numpy as np
from router_data import split, featurize, tokens
from overfit import sigmoid, log_loss, train

# 3. Which words does L1 keep? (one split, λ = 0.008)
((Xtr, ytr), _, _), vocab = featurize(*split(seed=0))
w, b = train(Xtr, ytr, 0.008, "l1")
kept = sorted((w[j], vocab[j]) for j in np.flatnonzero(w))
print(f"{len(kept)} of {len(vocab)} weights are nonzero; those of size >= 0.01:")
print("toward serving:", ", ".join(f"{t} {v:+.2f}" for v, t in kept[::-1] if v >= 0.01))
print("away:          ", ", ".join(f"{t} {v:+.2f}" for v, t in kept if v <= -0.01))