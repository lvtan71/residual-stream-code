import numpy as np

# Why distillation multiplies its loss by tau^2. With softened teacher
# p = softmax(t / tau) and student q = softmax(s / tau), the gradient of the
# soft cross-entropy with respect to the student's logits is (q - p) / tau.
# For large tau, q - p itself shrinks like 1 / tau, so the gradient shrinks
# like 1 / tau^2 (Hinton et al., 2015): (s - t) / (N tau^2) for zero-mean
# logits over N classes. Multiplying the loss by tau^2 undoes that, so the
# soft term's weight against the hard-label term doesn't depend on tau.
rng = np.random.default_rng(0)
N = 10
t, s = rng.normal(0, 2, N), rng.normal(0, 2, N)
t, s = t - t.mean(), s - s.mean()
softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
print(f"{'tau':>4s} {'|gradient|':>11s} {'x tau^2':>9s}   {'large-tau formula':>17s}")
for tau in (1, 2, 4, 8, 16):
    g = (softmax(s / tau) - softmax(t / tau)) / tau
    approx = (s - t) / (N * tau ** 2)
    print(f"{tau:4d} {np.linalg.norm(g):11.5f} {np.linalg.norm(g) * tau ** 2:9.4f}   {np.linalg.norm(approx):17.5f}")