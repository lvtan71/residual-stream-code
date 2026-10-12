"""Linear attention three ways, checked to give the same output.

1. quadratic: build the n x n weights, as softmax attention does
2. linear:    reassociate, phi(Q) (phi(K)^T V), never forming n x n
3. recurrent: one token at a time with a d x d state (causal case)
"""

import numpy as np

rng = np.random.default_rng(0)
n, d = 512, 64
Q, K, V = (rng.standard_normal((n, d)) for _ in range(3))


def phi(x):
    """elu(x) + 1, the feature map of Katharopoulos et al.: positive everywhere."""
    return np.where(x > 0, x + 1, np.exp(np.minimum(x, 0)))


def quadratic(Q, K, V, causal):
    A = phi(Q) @ phi(K).T                          # n x n similarities
    if causal:
        A = np.tril(A)
    return (A @ V) / A.sum(1, keepdims=True)


def linear(Q, K, V):
    S = phi(K).T @ V                               # d x d, shared by every query
    z = phi(K).sum(0)                              # d
    return (phi(Q) @ S) / (phi(Q) @ z)[:, None]


def recurrent(Q, K, V):
    s, z = np.zeros((d, d)), np.zeros(d)           # the whole memory
    out = np.empty_like(V)
    for i in range(len(Q)):
        k = phi(K[i])
        s += np.outer(k, V[i])                     # write token i into the state
        z += k
        q = phi(Q[i])
        out[i] = (q @ s) / (q @ z)                 # read with query i
    return out


print(f"n = {n}, d = {d}")
print(f"non-causal, quadratic vs linear:   max difference "
      f"{np.abs(quadratic(Q, K, V, False) - linear(Q, K, V)).max():.1e}")
print(f"causal, quadratic vs recurrent:    max difference "
      f"{np.abs(quadratic(Q, K, V, True) - recurrent(Q, K, V)).max():.1e}")

# exp(q.k) does not factor through a finite phi: its similarity matrix has
# full rank, while phi(Q) phi(K)^T has rank at most d.
E = np.exp(Q @ K.T / np.sqrt(d))
F = phi(Q) @ phi(K).T
print(f"rank of exp(QK^T / sqrt d): {np.linalg.matrix_rank(E)} of {n}")
print(f"rank of phi(Q) phi(K)^T:    {np.linalg.matrix_rank(F)} of {n}")
