import itertools
import numpy as np

# RLHF end to end on a space small enough to enumerate: every "response" is
# 3 tokens from a vocabulary of 8, 512 in all. A hidden gold reward stands
# in for what people actually prefer. People never report it; they compare
# two responses and prefer one with probability sigmoid(gold_a - gold_b),
# the Bradley-Terry model.
#   1. Reward model: fit a proxy reward to 300 comparisons of responses
#      sampled from the starting policy, by maximizing the Bradley-Terry
#      likelihood. It sees one feature per (position, token), so it can't
#      represent anything that depends on two tokens together.
#   2. Policy: the KL-regularized optimum, pi(y) ~ pi_ref(y) exp(r(y) / beta)
#      (derived in the post), computed exactly for a range of beta.
# Two worlds, the same except for one term in the gold reward.
#   "random": gold = a score per (position, token) + effects of some token
#     pairs, of either sign: errors the proxy can't represent, but with no
#     direction to them;
#   "in moderation": the same, plus one token (7) that is good once and bad
#     in excess, +b for one occurrence, 0 for two, -4 for three, the way
#     detail or length is in a real answer. The starting policy rarely
#     writes it twice, so comparisons teach the proxy "7 is good" at every
#     position. Two strengths, b = 2 and b = 4.
# 100 random worlds each.
V, L = 8, 3
Y = np.array(list(itertools.product(range(V), repeat=L)))          # (512, 3)
X = np.zeros((len(Y), L * V))
X[np.arange(len(Y))[:, None], np.arange(L) * V + Y] = 1             # one-hot per position
sigmoid = lambda z: 1 / (1 + np.exp(-z))

def world(kind, seed, bonus=0.0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, L * V)
    pair = rng.normal(0, 2.5, (V, V)) * (rng.random((V, V)) < 0.15)
    gold = X @ w + pair[Y[:, 0], Y[:, 2]]
    if kind == "in moderation":
        gold = gold + np.array([0.0, bonus, 0.0, -4.0])[(Y == 7).sum(1)]
    tok = np.exp(rng.normal(0, 0.5, (L, V)))
    tok /= tok.sum(1, keepdims=True)
    ref = tok[0, Y[:, 0]] * tok[1, Y[:, 1]] * tok[2, Y[:, 2]]
    return rng, gold, ref

def fit_reward_model(rng, gold, ref, n=300, steps=3000, lr=0.5, l2=1e-3):
    a, b = rng.choice(len(Y), n, p=ref), rng.choice(len(Y), n, p=ref)
    prefer_a = rng.random(n) < sigmoid(gold[a] - gold[b])          # noisy human labels
    win, lose = np.where(prefer_a, a, b), np.where(prefer_a, b, a)
    D = X[win] - X[lose]
    theta = np.zeros(L * V)
    for _ in range(steps):                                          # gradient ascent on the BT log-likelihood
        p = sigmoid(D @ theta)
        theta += lr * (D.T @ (1 - p) / n - l2 * theta)
    return X @ theta

betas = [3, 1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
for kind, bonus in (("random", 0.0), ("in moderation", 2.0), ("in moderation", 4.0)):
    rows = []                                                        # (seed, beta, KL, proxy gain, gold gain, 7s)
    for seed in range(100):
        rng, gold, ref = world(kind, seed, bonus)
        proxy = fit_reward_model(rng, gold, ref)
        for beta in betas:
            logits = np.log(ref) + proxy / beta
            logpi = logits - logits.max() - np.log(np.exp(logits - logits.max()).sum())
            pi = np.exp(logpi)
            rows.append((seed, beta, pi @ (logpi - np.log(ref)), pi @ proxy - ref @ proxy, pi @ gold - ref @ gold,
                         pi @ (Y == 7).sum(1)))
    rows = np.array(rows)
    G = rows[:, 4].reshape(100, len(betas))                         # gold gain, per world and beta
    fell = np.mean(G.max(1) - G[:, -1] > 0.5)
    name = kind if kind == "random" else f"{kind}, +{bonus:g} for one 7"
    print(f"\nworld \"{name}\": means over 100 worlds; the starting policy writes "
          f"{np.mean([world(kind, s, bonus)[2] @ (Y == 7).sum(1) for s in range(100)]):.2f} sevens per answer")
    print(f"{'beta':>6s} {'KL from ref':>12s} {'proxy gain':>11s} {'gold gain':>10s} {'7s per answer':>14s}")
    for beta in betas:
        r = rows[rows[:, 1] == beta]
        print(f"{beta:6g} {r[:, 2].mean():12.2f} {r[:, 3].mean():11.2f} {r[:, 4].mean():10.2f} {r[:, 5].mean():14.2f}")
    print(f"worlds whose gold gain ends more than 0.5 below its peak: {fell:.0%}")
