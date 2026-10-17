import numpy as np

def H(p):                  # entropy, in nats
    p = np.asarray(p, float)
    return -(p[p > 0] * np.log(p[p > 0])).sum()

def CE(p, q):              # cross-entropy of model q on true distribution p
    p, q = np.asarray(p, float), np.asarray(q, float)
    return -(p[p > 0] * np.log(q[p > 0])).sum()

truth = [0.7, 0.1, 0.1, 0.1]     # what the answer really is, for some question
for name, q in {"matches the truth": truth,
                "uniform guess":     [0.25] * 4,
                "overconfident":     [0.97, 0.01, 0.01, 0.01],
                "confidently wrong": [0.01, 0.97, 0.01, 0.01]}.items():
    ce = CE(truth, q)
    print(f"{name:18s} CE {ce:.3f} = H {H(truth):.3f} + KL {ce - H(truth):.3f}   "
          f"perplexity {np.exp(ce):.2f}")

print(f"\nuniform over 4 routes: ln 4 = {np.log(4):.3f} nats; "
      f"over a 128,256-token vocabulary: {np.log(128256):.2f} nats")
print(f"val CE 0.874 -> perplexity {np.exp(0.874):.2f}")

# Label smoothing: the best a model can do is to predict the smoothed target,
# which caps the gap between the right logit and the others.
for eps in (0.1, 0.01):
    top, rest = 1 - eps + eps / 4, eps / 4
    print(f"smoothing {eps}: target ({top:.4f}, {rest:.4f} x3), logit gap ln({top:.4f}/{rest:.4f}) "
          f"= {np.log(top / rest):.2f}, loss floor {H([top, rest, rest, rest]):.3f}")