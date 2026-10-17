import numpy as np
from a_loss import words as x, tokens as y

# "Learning Is Optimization"'s raw-feature fit: tokens = w * words + b, MSE loss, kappa = 128.
X = np.c_[x, np.ones_like(x)]
H = 2 / len(x) * X.T @ X                              # the Hessian
lo, hi = np.linalg.eigvalsh(H)
theta_star = np.linalg.solve(X.T @ X, X.T @ y)
grad = lambda th: 2 / len(x) * X.T @ (X @ th - y)
print(f"eigenvalues {lo:.4f} and {hi:.2f}, kappa {hi / lo:.0f}")

def steps_to(update, tol=1e-6, limit=100_000):
    """Steps until the parameters are within tol (relative) of the least-squares fit."""
    th, state = np.zeros(2), {}
    for t in range(1, limit + 1):
        th = update(th, state, t)
        if np.linalg.norm(th - theta_star) < tol * np.linalg.norm(theta_star):
            return t
    return None

def gd(eta):
    return lambda th, s, t: th - eta * grad(th)

def heavy_ball(eta, beta):
    def step(th, s, t):
        s["v"] = beta * s.get("v", 0) + grad(th)     # velocity: a running sum of gradients
        return th - eta * s["v"]
    return step

k = hi / lo
eta_gd = 2 / (hi + lo)                                # the best fixed step for plain descent
eta_hb = 4 / (np.sqrt(hi) + np.sqrt(lo)) ** 2         # Polyak's best choice for momentum
beta_hb = ((np.sqrt(k) - 1) / (np.sqrt(k) + 1)) ** 2
print(f"gradient descent, eta {eta_gd:.4f}:            {steps_to(gd(eta_gd)):5d} steps   "
      f"(theory: error shrinks by {(k - 1) / (k + 1):.4f} per step)")
print(f"momentum, eta {eta_hb:.4f}, beta {beta_hb:.3f}:   {steps_to(heavy_ball(eta_hb, beta_hb)):5d} steps   "
      f"(theory: {(np.sqrt(k) - 1) / (np.sqrt(k) + 1):.4f} per step)")
for beta in (0.5, 0.9, 0.99):
    n = steps_to(heavy_ball(eta_gd * (1 - beta), beta))
    print(f"momentum, beta {beta}, same effective step:   {n if n else 'no convergence':>5} steps")
# Tuned this way, the extreme directions sit exactly on the edge of the band,
# where the two roots coincide and the error decays like t * sqrt(beta)^t.
r, t = np.sqrt(beta_hb), np.arange(1, 1000)
print(f"first t with sqrt(beta)^t < 1e-6: {t[r ** t < 1e-6][0]}; with t * sqrt(beta)^t < 1e-6: "
      f"{t[t * r ** t < 1e-6][0]}")
