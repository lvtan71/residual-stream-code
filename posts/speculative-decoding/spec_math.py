import numpy as np

# 1. The acceptance rule makes the output exactly the target's distribution,
#    whatever the draft. Target p and draft q over 6 tokens: sample from q,
#    accept with probability min(1, p/q), otherwise sample from the residual
#    max(0, p - q), renormalized. Check by sampling a million times.
rng = np.random.default_rng(0)
p = np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
q = np.array([0.10, 0.10, 0.30, 0.30, 0.10, 0.10])           # a bad draft, on purpose
n = 1_000_000
x = rng.choice(6, n, p=q)
accept = rng.random(n) < np.minimum(1, p[x] / q[x])
resid = np.maximum(p - q, 0); resid /= resid.sum()
x = np.where(accept, x, rng.choice(6, n, p=resid))
print("target p:          ", np.round(p, 3))
print("output frequencies:", np.round(np.bincount(x, minlength=6) / n, 3))
print(f"acceptance rate {accept.mean():.3f}; theory: sum of min(p, q) = {np.minimum(p, q).sum():.3f}")

# 2. Tokens per verification step. If each drafted token is accepted with
#    probability alpha, independently, a step with k drafts yields
#    (1 - alpha^(k+1)) / (1 - alpha) tokens on average: the accepted run,
#    plus the one token the target always contributes (a correction, or a
#    bonus token after k acceptances). Check against simulation.
def simulate(alpha, k, steps=200_000):
    acc = rng.random((steps, k)) < alpha
    run = np.where(acc.all(1), k, acc.argmin(1))              # accepted before the first rejection
    return (run + 1).mean()
print("\nalpha  k   formula  simulated")
for alpha in (0.6, 0.8, 0.9):
    for k in (2, 4, 8):
        f = (1 - alpha ** (k + 1)) / (1 - alpha)
        print(f"{alpha:5.1f} {k:2d} {f:9.2f} {simulate(alpha, k):10.2f}")

# 3. Speedup, counting the draft's cost: c = one draft step's time relative
#    to one target step. A step costs k draft steps plus one target step that
#    verifies all k (decode is memory-bound, so verifying k+1 positions costs
#    about one step).
print("\nspeedup over plain decoding, by draft cost c (alpha = 0.8)")
print("  k " + "".join(f"  c={c:<5g}" for c in (0.0, 0.05, 0.1, 0.3)))
for k in (1, 2, 4, 8, 16):
    f = (1 - 0.8 ** (k + 1)) / (1 - 0.8)
    print(f"{k:3d} " + "".join(f"{f / (k * c + 1):8.2f}" for c in (0.0, 0.05, 0.1, 0.3)))