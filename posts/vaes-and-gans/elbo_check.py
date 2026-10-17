import numpy as np

# A linear-Gaussian latent model in 2-D, small enough to solve exactly:
#   p(z) = N(0, I),   p(x | z) = N(W z + b, s^2 I).
# Then p(x) = N(b, W W^T + s^2 I), and the true posterior p(z | x) is Gaussian.
# For any Gaussian q(z | x) the ELBO is reconstruction - KL(q || prior), and
# log p(x) - ELBO = KL(q || true posterior). Checked here three ways.
rng = np.random.default_rng(0)
W, b, s = np.array([[1.0, 0.5], [-0.3, 0.8]]), np.array([0.2, -0.1]), 0.5
x = np.array([1.0, 0.4])

def log_normal(v, mean, cov):
    d = v - mean
    return -0.5 * (len(v) * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1] + d @ np.linalg.solve(cov, d))

def kl_gauss(m1, S1, m0, S0):                       # KL(N(m1, S1) || N(m0, S0))
    S0i = np.linalg.inv(S0)
    return 0.5 * (np.trace(S0i @ S1) + (m0 - m1) @ S0i @ (m0 - m1) - len(m1)
                  + np.linalg.slogdet(S0)[1] - np.linalg.slogdet(S1)[1])

def elbo(mu, sigma, n=200_000):
    """Reconstruction term by sampling (the reparameterization trick: z = mu +
    sigma * eps), KL to the prior in closed form."""
    z = mu + sigma * rng.standard_normal((n, 2))
    r = x - z @ W.T - b
    rec = (-0.5 * (r ** 2).sum(1) / s ** 2 - np.log(2 * np.pi * s ** 2)).mean()
    kl = 0.5 * (mu ** 2 + sigma ** 2 - 1 - np.log(sigma ** 2)).sum()
    r0 = x - W @ mu - b                               # the same expectation, in closed form
    exact = -0.5 * (r0 @ r0 + (W ** 2 * sigma ** 2).sum()) / s ** 2 - np.log(2 * np.pi * s ** 2)
    return rec, exact, kl

log_px = log_normal(x, b, W @ W.T + s ** 2 * np.eye(2))
P = np.linalg.inv(np.eye(2) + W.T @ W / s ** 2)      # posterior covariance
m = P @ W.T @ (x - b) / s ** 2                        # posterior mean
print(f"log p(x), exact: {log_px:.4f}")
print(f"true posterior: mean ({m[0]:.3f}, {m[1]:.3f}), "
      f"std ({np.sqrt(P[0, 0]):.3f}, {np.sqrt(P[1, 1]):.3f}), correlation {P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]):.3f}\n")
print(f"{'q(z|x)':36s} {'recon (sampled, exact)':>22s} {'KL to prior':>11s} {'ELBO':>8s} {'gap':>7s} {'KL to post':>10s}")
# The best diagonal q (smallest KL to the posterior) keeps the posterior's
# mean but takes std_j = 1 / sqrt(precision_jj), not the posterior's own
# marginal stds: Kingma and Welling's encoder can't do better than this.
best = 1 / np.sqrt(np.diag(np.linalg.inv(P)))
cases = [("best diagonal Gaussian", m, best),
         ("posterior's mean and marginal stds", m, np.sqrt(np.diag(P))),
         ("right mean, stds 3x too wide", m, 3 * np.sqrt(np.diag(P))),
         ("the prior, N(0, I)", np.zeros(2), np.ones(2))]
for name, mu, sigma in cases:
    rec, exact, kl = elbo(mu, sigma)
    print(f"{name:36s} {rec:10.4f} {exact:10.4f}  {kl:11.4f} {exact - kl:8.4f} {log_px - (exact - kl):7.4f} "
          f"{kl_gauss(mu, np.diag(sigma ** 2), m, P):10.4f}")

mu, sigma = np.array([1.0, -0.5]), np.array([0.5, 2.0])
print(f"\nKL(N(mu, sigma^2) || N(0, I)) for mu = (1, -0.5), sigma = (0.5, 2): "
      f"{0.5 * (mu ** 2 + sigma ** 2 - 1 - np.log(sigma ** 2)).sum():.4f} nats")