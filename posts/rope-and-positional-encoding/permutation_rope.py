import numpy as np

# 1. Attention without positions can't tell order apart: permute the tokens,
#    and every output is permuted the same way; nothing else changes. With a
#    causal mask that's no longer true: each token sees a different prefix.
rng = np.random.default_rng(0)
T, H = 5, 8
X = rng.normal(size=(T, H))
Wq, Wk, Wv = (rng.normal(size=(H, H)) / np.sqrt(H) for _ in range(3))
def attend(X, causal=False):
    s = (X @ Wq) @ (X @ Wk).T / np.sqrt(H)
    if causal:
        s = np.where(np.tri(T, dtype=bool), s, -np.inf)
    w = np.exp(s - s.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    return w @ (X @ Wv)
perm = rng.permutation(T)
print(f"attention(permuted input) == permuted attention(input): "
      f"{np.allclose(attend(X[perm]), attend(X)[perm])}")
print(f"the same, with a causal mask: "
      f"{np.allclose(attend(X[perm], causal=True), attend(X, causal=True)[perm])}")

# 2. RoPE: rotate each pair of dimensions (2i, 2i+1) by angle pos * theta_i.
def rope(x, pos, base=10_000.0):
    half = x.shape[-1] // 2
    theta = base ** (-np.arange(half) / half)             # one frequency per pair
    ang = pos * theta
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * np.cos(ang) - x2 * np.sin(ang)
    out[..., 1::2] = x1 * np.sin(ang) + x2 * np.cos(ang)
    return out

q, k = rng.normal(size=H), rng.normal(size=H)
for m, n in [(3, 5), (10, 12), (100, 102), (3, 10)]:
    print(f"query at {m:3d}, key at {n:3d} (distance {n - m:2d}): score {rope(q, m) @ rope(k, n):+.4f}")
print(f"rotation keeps length: |q| = {np.linalg.norm(q):.4f}, |rope(q, 57)| = {np.linalg.norm(rope(q, 57)):.4f}")