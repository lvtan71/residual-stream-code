import numpy as np

# Attention one block of keys at a time, never holding a full row of scores.
# The running state per query row: m, the largest score seen so far; l, the
# sum of exp(score - m) so far; o, the weighted sum of values so far, also
# relative to m. When a new block raises the max from m to m', everything
# accumulated is rescaled by exp(m - m') (Milakov and Gimelshein, 2018).
rng = np.random.default_rng(0)

def attention(Q, K, V):
    S = Q @ K.T / np.sqrt(Q.shape[1])                     # the full n x n score matrix
    P = np.exp(S - S.max(1, keepdims=True))
    return (P / P.sum(1, keepdims=True)) @ V

def tiled_attention(Q, K, V, block):
    n, d = Q.shape
    m = np.full(n, -np.inf)                               # running max per row
    l = np.zeros(n)                                       # running sum of exponentials
    o = np.zeros((n, V.shape[1]))                         # running (unnormalized) output
    for j in range(0, K.shape[0], block):
        S = Q @ K[j:j + block].T / np.sqrt(d)             # only n x block scores at a time
        m_new = np.maximum(m, S.max(1))
        scale = np.exp(m - m_new)                         # how much the old max overstated
        P = np.exp(S - m_new[:, None])
        l = l * scale + P.sum(1)
        o = o * scale[:, None] + P @ V[j:j + block]
        m = m_new
    return o / l[:, None]

# The worked example: one query, four keys, two blocks of two.
q = np.array([[1.0, 0.0]])
k = np.array([[2.0, 0.0], [1.0, 0.0], [4.0, 0.0], [0.0, 0.0]]) * np.sqrt(2)   # scores 2, 1, 4, 0
v = np.array([[1.0], [2.0], [3.0], [4.0]])
s = (q @ k.T / np.sqrt(2))[0]
print("scores", s, "-> softmax", np.round(np.exp(s - s.max()) / np.exp(s - s.max()).sum(), 4))
m, l, o = -np.inf, 0.0, 0.0
for j, blk in enumerate(((0, 1), (2, 3))):
    sb = s[list(blk)]
    m_new = max(m, sb.max())
    scale = np.exp(m - m_new)
    p = np.exp(sb - m_new)
    l, o, m = l * scale + p.sum(), o * scale + (p * v[list(blk), 0]).sum(), m_new
    print(f"after block {j + 1}: max {m:.0f}, sum {l:.4f}, output so far {o:.4f} -> {o / l:.4f}")
print(f"standard attention gives {attention(q, k, v)[0, 0]:.4f}")

for n, d, block in ((1024, 64, 128), (4096, 128, 256)):
    Q, K, V = (rng.normal(size=(n, d)) for _ in range(3))
    err = np.abs(tiled_attention(Q, K, V, block) - attention(Q, K, V)).max()
    print(f"n = {n}, d = {d}, blocks of {block}: largest difference from standard attention {err:.1e}")