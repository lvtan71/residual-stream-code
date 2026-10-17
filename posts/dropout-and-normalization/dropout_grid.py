import sys
import numpy as np
from router_data import split
from router4 import four_way
from net import init, fit, forward
from softmax_basics import softmax, cross_entropy
from optim import Adam

# A two-hidden-layer router that can memorize its 120 training questions.
# Does dropout help it, alone or with AdamW's weight decay ("Initialization and Optimizers")?
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
CONFIGS = [(p, lam) for lam in (0.0, 0.3, 1.0, 3.0, 10.0) for p in (0.0, 0.1, 0.3, 0.5, 0.7)]
rows = {c: [] for c in CONFIGS}
for seed in range(20):
    ((Xtr, ytr), (Xva, yva), (Xte, yte)), _ = four_way(*split(seed))
    for p, lam in CONFIGS:
        rng = np.random.default_rng(seed)
        layers = init([Xtr.shape[1], 64, 64, 4], rng)
        fit(layers, Adam(1e-3, decay=lam), Xtr, ytr, STEPS, rng, drop=p)
        zv, zt = forward(layers, Xva)[0], forward(layers, Xte)[0]      # no dropout at test time
        top2 = np.sort(zv, 1)[:, -2:]                              # the two largest logits
        wrong = zv.argmax(1) != yva
        rows[(p, lam)].append((cross_entropy(zv, yva), (zv.argmax(1) == yva).mean(),
                               softmax(zv).max(1).mean(), cross_entropy(zt, yte),
                               (top2[:, 1] - top2[:, 0]).mean(), softmax(zv).max(1)[wrong].mean()))
corrected = np.sqrt(1 / 20 + 40 / 120)
se = lambda d: f"{d.mean():+.3f} ± {d.std(ddof=1) * corrected:.3f}"
base = np.array(rows[(0.0, 0.0)])
print(f"{STEPS:,} steps of Adam. Differences are paired, against no dropout at the same decay.")
for (p, lam), r in rows.items():
    r, same = np.array(r), np.array(rows[(0.0, lam)])
    print(f"dropout {p:<4} decay {lam:<4} val CE {r[:, 0].mean():.3f} "
          f"{se(r[:, 0] - same[:, 0]) if p else '':>15s}  val acc {r[:, 1].mean():.3f} "
          f"{se(r[:, 1] - same[:, 1]) if p else '':>15s}  mean top p {r[:, 2].mean():.3f}  "
          f"top-2 logit gap {r[:, 4].mean():5.2f}  top p when wrong {r[:, 5].mean():.3f}")
pick = lambda cs: min(cs, key=lambda c: np.mean(np.array(rows[c])[:, 0]))   # on validation
best, decay_only = pick(CONFIGS), pick([c for c in CONFIGS if c[0] == 0.0])
t, d = np.array(rows[best])[:, 3], np.array(rows[decay_only])[:, 3]
print(f"chosen on validation: dropout {best[0]}, decay {best[1]}; without dropout: decay {decay_only[1]}")
print(f"test CE: {t.mean():.3f} chosen, {d.mean():.3f} decay only, {base[:, 3].mean():.3f} neither. "
      f"Dropout's own share: {se(t - d)}; decay's: {se(d - base[:, 3])}")
