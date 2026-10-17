import numpy as np
from router_data import english, split
from router4 import four_way
from norms import batchnorm, layernorm

rng = np.random.default_rng(0)
tr, va, te = split(0)
((X, y), (Xall, _)), _ = four_way(tr, english)
W = rng.normal(0, np.sqrt(2 / X.shape[1]), (X.shape[1], 64))   # one He-initialized layer
Z = Xall @ W                                                     # its pre-activations, all 200 questions
g, b = np.ones(64), np.zeros(64)

# 1. The same question, normalized inside two different batches of 16.
others = rng.permutation(np.arange(1, 200))
out1 = batchnorm(Z[np.r_[0, others[:15]]], g, b)[0][0]
out2 = batchnorm(Z[np.r_[0, others[15:30]]], g, b)[0][0]
print(f"one question, two batches: its BatchNorm outputs differ by {np.abs(out1 - out2).mean():.2f} "
      f"on average (their typical size is {np.abs(out1).mean():.2f})")

# 2. How far a batch's mean is from the mean over all 200 questions, by batch size.
full_mu, full_sd = Z.mean(0), Z.std(0)
for B in (2, 8, 32, 128):
    gaps = [np.abs(Z[rng.choice(200, B, replace=False)].mean(0) - full_mu).mean() / full_sd.mean()
            for _ in range(500)]
    print(f"batch of {B:3d}: batch mean is off by {np.mean(gaps):.2f} standard deviations on average")

# 3. At inference, a batch of one: every feature equals its own mean.
print(f"batch of 1: BatchNorm output {np.abs(batchnorm(Z[:1], g, b)[0]).max():.1f} everywhere; "
      f"LayerNorm output still varies, size {np.abs(layernorm(Z[:1], g, b)[0]).mean():.2f}")

# 4. A sequence: normalize 8 token vectors together (over positions), then change
# only the *last* token. Does the *first* token's output change?
seq = rng.normal(size=(8, 64))
later = seq.copy(); later[-1] += 3.0
bn_gap = np.abs(batchnorm(seq, g, b)[0][0] - batchnorm(later, g, b)[0][0]).max()
ln_gap = np.abs(layernorm(seq, g, b)[0][0] - layernorm(later, g, b)[0][0]).max()
print(f"change the last token only: first token's output moves by {bn_gap:.3f} with BatchNorm, "
      f"{ln_gap:.3f} with LayerNorm")
