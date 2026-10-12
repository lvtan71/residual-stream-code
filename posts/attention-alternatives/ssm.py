"""State space models: the recurrence, the convolution, the scan and the duality.

One input channel, a state of N numbers, diagonal A, as in S4D and Mamba.
Every check compares two ways of computing the same outputs.
"""

import numpy as np

rng = np.random.default_rng(0)
T, N = 256, 16                                   # sequence length, state size
A = -np.exp(rng.standard_normal(N))              # negative: a decaying state
x = rng.standard_normal(T)


def discretize(delta, A, B):
    """Zero-order hold for diagonal A: Abar = exp(delta A), Bbar = (exp(delta A) - 1) / A * B."""
    Abar = np.exp(delta * A)
    return Abar, (Abar - 1) / A * B


def recurrence(Abar, Bbar, C, x):
    """h_t = Abar_t * h_{t-1} + Bbar_t x_t,  y_t = C_t . h_t; parameters may change with t."""
    h, y = np.zeros(N), np.empty(len(x))
    for t in range(len(x)):
        h = Abar[t] * h + Bbar[t] * x[t]
        y[t] = C[t] @ h
    return y


print("delta is a gate: Abar = exp(delta A) for this model's slowest and fastest state")
for delta in (0.001, 0.1, 10.0):
    Abar = np.exp(delta * A)
    print(f"  delta = {delta:<6}  keeps {Abar.max():.3f} and {Abar.min():.3f} of the old state")

# S4: fixed delta, B, C. The recurrence is then a convolution with kernel K_k = C Abar^k Bbar.
delta, B, C = 0.1, rng.standard_normal(N), rng.standard_normal(N)
Abar, Bbar = discretize(delta, A, B)
y_rec = recurrence([Abar] * T, [Bbar] * T, [C] * T, x)
kernel = np.array([C @ (Abar ** k * Bbar) for k in range(T)])
y_conv = np.convolve(x, kernel)[:T]
print(f"\nfixed parameters, recurrence vs convolution: max difference {np.abs(y_rec - y_conv).max():.1e}")

# Mamba: delta, B, C computed from each input. No single kernel describes the system any more.
W_d, W_B, W_C = rng.standard_normal(), rng.standard_normal(N), rng.standard_normal(N)
deltas = np.log1p(np.exp(W_d * x))               # softplus, one step size per token
Bs, Cs = np.outer(x, W_B), np.outer(x, W_C)      # one B and one C per token
Abars, Bbars = zip(*(discretize(dt, A, b) for dt, b in zip(deltas, Bs)))
y_sel = recurrence(Abars, Bbars, Cs, x)
kernel0 = np.array([Cs[0] @ (Abars[0] ** k * Bbars[0]) for k in range(T)])
print(f"input-dependent, recurrence vs first token's kernel: max difference "
      f"{np.abs(y_sel - np.convolve(x, kernel0)[:T]).max():.1e}")


# The selective scan: h_t = a_t h_{t-1} + b_t composes associatively,
# (a1, b1) then (a2, b2) = (a1 a2, a2 b1 + b2), so it runs in log2 T parallel steps.
def scan(a, b):
    a, b, steps, k = a.copy(), b.copy(), 0, 1
    while k < len(a):
        b[k:] = a[k:] * b[:-k] + b[k:]           # every position combines with the one k back
        a[k:] = a[k:] * a[:-k]
        k, steps = 2 * k, steps + 1
    return b, steps


a = np.array(Abars)                              # T x N
b = np.array(Bbars) * x[:, None]
H, steps = scan(a, b)
print(f"parallel scan in {steps} steps vs {T} sequential steps: max difference "
      f"{np.abs((H * Cs).sum(1) - y_sel).max():.1e}")

# Mamba-2: make A a scalar per step, a_t. The whole layer is then one masked matrix,
# y = (L * (C B^T)) x with L[t, s] = a_{s+1} ... a_t: attention's quadratic form.
a_t = np.exp(-deltas)                            # one scalar decay per token
Bk, Cq = rng.standard_normal((T, N)), rng.standard_normal((T, N))
h, y_lin = np.zeros(N), np.empty(T)
for t in range(T):
    h = a_t[t] * h + Bk[t] * x[t]
    y_lin[t] = Cq[t] @ h
logs = np.cumsum(np.log(a_t))
L = np.tril(np.exp(logs[:, None] - logs[None, :]))
y_quad = (L * (Cq @ Bk.T)) @ x
print(f"scalar decay, linear recurrence vs quadratic masked form: max difference "
      f"{np.abs(y_lin - y_quad).max():.1e}")
print("with every a_t = 1, L is the causal mask and (L * (C B^T)) x is causal "
      "linear attention, Q = C, K = B, V = x")
