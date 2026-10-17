import numpy as np
from router_data import split, featurize, tokens, vietnamese
from overfit import sigmoid, log_loss, train

SERVING = {"kv", "cache", "caching", "batch", "batching", "latency",
           "throughput", "serve", "serving", "vllm", "sglang", "tensorrt",
           "gpu", "decode", "prefill", "paged", "quantization", "memory"}

def keyword_count(rs):
    return np.array([[sum(t in SERVING for t in tokens(r["question"]))]
                     for r in rs], dtype=float)

def fit_all(seed):
    """Train the three routers of "Features and Overfitting" on one split; return logits."""
    tr, va, te = split(seed)
    ((Xtr, ytr), (Xva, yva), (Xte, yte), (Xvi, yvi)), vocab = featurize(
        tr, va, te, vietnamese)
    kw = [keyword_count(rs) for rs in (tr, va, te, vietnamese)]
    designs = {
        "no penalty": ([Xtr, Xva, Xte, Xvi], 0.0),
        "L2": ([Xtr, Xva, Xte, Xvi], 0.003),
        "L2 + keywords": ([np.c_[X, k] for X, k in zip([Xtr, Xva, Xte, Xvi], kw)], 0.003),
    }
    out = {}
    for name, (mats, lam) in designs.items():
        w, b = train(mats[0], ytr, lam)
        out[name] = {part: (mats[i] @ w + b, y) for i, (part, y) in
                     enumerate([("train", ytr), ("val", yva), ("test", yte), ("vi", yvi)])}
    return out

RUNS = [fit_all(seed) for seed in range(20)]

def pooled(name, part):
    """Logits and labels for one model and one part, pooled over 20 splits."""
    z = np.concatenate([run[name][part][0] for run in RUNS])
    y = np.concatenate([run[name][part][1] for run in RUNS])
    return z, y