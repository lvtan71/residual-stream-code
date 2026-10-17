import numpy as np
from overfit import sigmoid
from router_data import split, featurize, tokens, vietnamese
from router_models import RUNS, pooled

# How much of each Vietnamese question does the English vocabulary see?
(train_set, *_), vocab = featurize(*split(seed=0))
z, y = RUNS[0]["L2 + keywords"]["vi"]
for r, p in zip(vietnamese, sigmoid(z)):
    if r["route"] == "serving":
        seen = [t for t in tokens(r["question"]) if t in vocab]
        print(f"p={p:.2f} {'caught' if p >= 0.5 else 'MISSED'}  seen={seen}  {r['question']}")

# Over all 20 splits: how often is each serving question caught, and how
# often is a non-serving question wrongly sent to serving?
for name in ["no penalty", "L2", "L2 + keywords"]:
    caught = np.mean([sigmoid(run[name]["vi"][0]) >= 0.5 for run in RUNS], axis=0)
    z, y = pooled(name, "vi")
    recall = ((sigmoid(z) >= 0.5) & (y == 1)).sum() / (y == 1).sum()
    false_alarm = ((sigmoid(z) >= 0.5) & (y == 0)).sum() / (y == 0).sum()
    print(f"{name:14s} recall {recall:.3f}  false-alarm rate {false_alarm:.3f}"
          f"  per-question catch rate: {' '.join(f'{c:.2f}' for c in caught[:10])}")