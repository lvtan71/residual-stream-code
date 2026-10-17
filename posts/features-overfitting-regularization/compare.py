import numpy as np
from router_data import split, featurize, tokens
from overfit import sigmoid, log_loss, train

# 4. The hand-picked keyword count of "Learning Is Optimization" vs the learned bag of words.
SERVING = {"kv", "cache", "caching", "batch", "batching", "latency",
           "throughput", "serve", "serving", "vllm", "sglang", "tensorrt",
           "gpu", "decode", "prefill", "paged", "quantization", "memory"}
def hits(rs, words=SERVING):
    return np.array([[sum(t in words for t in tokens(r["question"]))]
                     for r in rs], dtype=float)
scores = {"keyword count": [], "keyword count, seen words only": [],
          "bag of words": [], "both": []}
for seed in range(20):
    tr, va, te = split(seed)
    ((Xtr, ytr), (Xva, yva), _), vocab = featurize(tr, va, te)
    seen = SERVING & set(vocab)             # keywords that occur in training
    designs = {"keyword count": (hits(tr), hits(va), 0.0),
               "keyword count, seen words only": (hits(tr, seen), hits(va, seen), 0.0),
               "bag of words": (Xtr, Xva, 0.003),
               "both": (np.c_[Xtr, hits(tr)], np.c_[Xva, hits(va)], 0.003)}
    for name, (A, B, lam) in designs.items():
        w, b = train(A, ytr, lam)
        p = sigmoid(B @ w + b)
        scores[name].append((log_loss(p, yva), ((p >= 0.5) == yva).mean()))
for name, s in scores.items():
    s = np.array(s)
    print(f"{name:31s} val loss {s[:, 0].mean():.3f}  val acc {s[:, 1].mean():.3f}")