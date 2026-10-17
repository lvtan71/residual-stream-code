import numpy as np

# Arena-style ratings: people compare two anonymous models' answers and vote
# for one. The Bradley-Terry model ("RLHF") says model i beats model j with
# probability sigmoid(r_i - r_j); fit the ratings r by maximum likelihood
# and report them on the Elo-like scale arenas use (400 / ln 10 per unit).
# Four models with known true ratings; votes on random pairs; 95% intervals
# from 200 bootstrap resamples of the votes.
rng = np.random.default_rng(0)
true = np.array([0.0, 0.10, 0.20, 0.60])               # in logits: small, small, small, large gaps
names = ["model A", "model B", "model C", "model D"]
SCALE = 400 / np.log(10)

def fit(i, j, win, iters=500, lr=0.5):
    """Maximum likelihood by gradient ascent; model A pinned at 0."""
    r = np.zeros(len(true))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(r[i] - r[j])))
        g = np.zeros_like(r)
        np.add.at(g, i, win - p); np.add.at(g, j, p - win)
        r += lr * g / len(win) * len(true)
        r -= r[0]
    return r

for votes in (500, 5000, 50000):
    i = rng.integers(0, 4, votes); j = (i + rng.integers(1, 4, votes)) % 4
    win = (rng.random(votes) < 1 / (1 + np.exp(-(true[i] - true[j])))).astype(float)
    r = fit(i, j, win)
    boot = []
    for _ in range(200):
        k = rng.integers(0, votes, votes)
        boot.append(fit(i[k], j[k], win[k]))
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    print(f"{votes:,} votes:")
    for n, t, est, a, b in zip(names, true, r, lo, hi):
        print(f"  {n}: true {t * SCALE:5.0f}, estimated {est * SCALE:5.0f}, 95% interval {a * SCALE:5.0f} to {b * SCALE:5.0f}")