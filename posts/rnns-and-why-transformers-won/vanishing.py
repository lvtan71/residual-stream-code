import numpy as np

# How much of the gradient at step 50 survives back to step 0?
# Backprop multiplies by W_h^T (times tanh's slope) once per step, 50 times.
rng = np.random.default_rng(0)
d = 64
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))          # a rotation: every direction the same size

def run(radius, act, input_size=0.5):
    W = radius * Q                                     # every singular value equals `radius`
    h, hs = rng.normal(size=d) * 0.5, []
    for _ in range(50):                                # forward, keeping the states
        z = h @ W + rng.normal(size=d) * input_size    # plus a fresh input each step
        h = np.tanh(z) if act == "tanh" else z
        hs.append(h)
    g, sizes = np.ones(d) / np.sqrt(d), {}
    for t in range(49, -1, -1):                        # backward from step 50
        if act == "tanh":
            g = g * (1 - hs[t] ** 2)
        g = g @ W.T
        if 50 - t in (1, 10, 25, 50):
            sizes[50 - t] = np.linalg.norm(g)
    return sizes

for radius in (0.9, 1.0, 1.1, 2.0):
    for act in ("linear", "tanh"):
        if act == "linear" and radius == 2.0:
            continue                                   # 2^50: no need to print it
        s = run(radius, act)
        print(f"radius {radius}, {act:6s}: gradient size after " +
              ", ".join(f"{k} steps {v:.1e}" for k, v in s.items()))
s = run(1.1, "tanh", input_size=0.0)
print("radius 1.1, tanh, no inputs: gradient size after " + ", ".join(f"{k} steps {v:.1e}" for k, v in s.items()))