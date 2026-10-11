import numpy as np

# Generating 4 tokens with the post's toy projections, recomputing every
# key and value each step (no cache) and computing only the new token's
# (cached): the outputs match, and the work differs.

def attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    w = np.exp(scores - scores.max(axis=-1, keepdims=True))
    w /= w.sum(axis=-1, keepdims=True)
    return w @ V

def fmt(v):
    return "[" + " ".join(f"{x:.3f}" for x in v) + "]"

W_Q = np.array([[1, 0], [1, 1]], dtype=float)
W_K = np.array([[0, 1], [1, 0]], dtype=float)
W_V = np.array([[1, 1], [0, 1]], dtype=float)
X = np.array([[1, 0], [0, 1], [1, -1], [0.5, 0.5]], dtype=float)

n = len(X)
K_cache, V_cache = [], []
naive_rows = cached_rows = 0

for t in range(1, n + 1):
    # naive: recompute K and V for all t tokens from scratch, every step
    Kt, Vt = X[:t] @ W_K, X[:t] @ W_V
    y_naive = attention(X[t - 1:t] @ W_Q, Kt, Vt)[0]
    naive_rows += t

    # cached: compute K and V for only the new token, append to the cache
    K_cache.append(X[t - 1] @ W_K)
    V_cache.append(X[t - 1] @ W_V)
    y_cached = attention(X[t - 1:t] @ W_Q, np.array(K_cache), np.array(V_cache))[0]
    cached_rows += 1

    print(f"step {t}  no-cache={fmt(y_naive)}  cached={fmt(y_cached)}  rows: {t} vs 1")

print(f"total K/V rows computed: {naive_rows} (= n(n+1)/2)  vs  {cached_rows} (= n)")

# Step 4 in detail: token 4's query against the whole cached K.
q4 = X[3] @ W_Q
scores = q4 @ np.array(K_cache).T / np.sqrt(2)
w = np.exp(scores - scores.max()); w /= w.sum()
print(f"step 4: q4 = {fmt(q4)}, scores q4 K^T/sqrt(2) = {fmt(scores)}, weights = {fmt(w)}, "
      f"output = {fmt(w @ np.array(V_cache))}")
