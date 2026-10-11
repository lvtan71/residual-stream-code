import numpy as np

# The three attention patterns as one function, on the post's small worked
# examples: self-attention over two tokens, causal (masked) self-attention
# over three, and cross-attention from a two-token decoder into a
# three-token encoder. Then a check that attention without positions is
# permutation-equivariant: shuffling the input rows shuffles the output rows
# the same way and changes nothing else.
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

def attention(Q, K, V, mask=None):
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    if mask is not None:
        scores = scores + mask                       # -inf where a query may not look
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)  # softmax, row by row
    return scores, weights, weights @ V

W_Q = np.array([[1, 0], [1, 1]], dtype=float)
W_K = np.array([[0, 1], [1, 0]], dtype=float)
W_V = np.array([[1, 1], [0, 1]], dtype=float)

def show(name, scores, weights, out):
    print(f"{name}\nscores QK^T/sqrt(d):\n{scores}\nweights A:\n{weights}\noutput AV:\n{out}\n")

X = np.array([[1, 0], [0, 1]], dtype=float)
show("self-attention, 2 tokens", *attention(X @ W_Q, X @ W_K, X @ W_V))

X3 = np.array([[1, 0], [0, 1], [1, -1]], dtype=float)
causal = np.triu(np.full((3, 3), -np.inf), k=1)     # -inf above the diagonal, 0 elsewhere
show("causal self-attention, 3 tokens", *attention(X3 @ W_Q, X3 @ W_K, X3 @ W_V, causal))

D = np.array([[0.5, 0.5], [-1, 1]], dtype=float)    # decoder states; X3 is the encoder's output
show("cross-attention, 2 decoder tokens into 3 encoder tokens", *attention(D @ W_Q, X3 @ W_K, X3 @ W_V))

rng = np.random.default_rng(0)
Xr = rng.standard_normal((6, 2))
perm = rng.permutation(6)
_, _, Y = attention(Xr @ W_Q, Xr @ W_K, Xr @ W_V)
_, _, Yp = attention(Xr[perm] @ W_Q, Xr[perm] @ W_K, Xr[perm] @ W_V)
print(f"6 random tokens, rows shuffled by {perm}: output rows shuffled the same way, "
      f"largest difference {np.abs(Yp - Y[perm]).max():.1e}")
