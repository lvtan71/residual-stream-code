import numpy as np

# An Elman RNN with a 2-number state, run for 3 steps:
#   h_t = tanh(x_t W_x + h_{t-1} W_h + b),   and a prediction y = h_3 w_out at the end.
W_x = np.array([[0.5, -0.3]])                # one input number per step
W_h = np.array([[0.8, 0.2], [-0.4, 0.6]])    # the same matrix at every step
b, w_out = np.array([0.1, 0.0]), np.array([1.0, -1.0])
xs = [1.0, 0.0, -1.0]

def forward(W_h):
    h, hs = np.zeros(2), [np.zeros(2)]
    for x in xs:
        h = np.tanh(x * W_x[0] + h @ W_h + b)
        hs.append(h)
    return hs, hs[-1] @ w_out

hs, y = forward(W_h)
for t, h in enumerate(hs[1:], 1):
    print(f"h_{t} = {np.round(h, 4)}")
print(f"prediction y = h_3 . w_out = {y:.4f}")

# Backprop through time: run "Backpropagation, by Hand"'s rules backward through the 3 steps.
# The same W_h was used at every step, so its gradient is the sum over steps.
dW_h, dh = np.zeros_like(W_h), w_out.copy()          # dy/dh_3
for t in range(3, 0, -1):
    dz = dh * (1 - hs[t] ** 2)                        # back through tanh
    dW_h += np.outer(hs[t - 1], dz)                   # this step's share of the gradient
    dh = dz @ W_h.T                                   # pass back to h_{t-1}
    print(f"gradient reaching h_{t - 1}: {np.round(dh, 4)}")
eps, numeric = 1e-6, np.zeros_like(W_h)
for i in np.ndindex(W_h.shape):
    E = np.zeros_like(W_h); E[i] = eps
    numeric[i] = (forward(W_h + E)[1] - forward(W_h - E)[1]) / (2 * eps)
print(f"dy/dW_h = {np.round(dW_h, 4).tolist()}; largest gap to finite differences "
      f"{np.abs(dW_h - numeric).max():.1e}")