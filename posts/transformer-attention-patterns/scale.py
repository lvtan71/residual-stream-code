import numpy as np

# Why attention divides by sqrt(d). With query and key entries independent,
# mean 0 and variance 1, a dot product's spread grows as sqrt(d). Measured
# over 20,000 random pairs per width, then what that does to a softmax over
# 16 keys: without scaling, the largest weight heads toward 1 and the
# softmax's gradient (the norm of its Jacobian, diag(p) - p p^T) toward 0.
rng = np.random.default_rng(0)
print("   d   std of q.k   std of q.k/sqrt(d)   softmax over 16 keys:  largest weight   gradient norm")
print("                                                               raw    scaled     raw     scaled")
for d in (4, 16, 64, 256, 1024):
    q, k = rng.standard_normal((20_000, d)), rng.standard_normal((20_000, d))
    dots = (q * k).sum(1)
    rows = {}
    for name, s in (("raw", 1.0), ("scaled", np.sqrt(d))):
        top, grad = [], []
        for _ in range(2_000):
            z = np.einsum("d,dk->k", rng.standard_normal(d), rng.standard_normal((d, 16))) / s  # 1 query, 16 keys
            p = np.exp(z - z.max()); p /= p.sum()
            top.append(p.max())
            grad.append(np.linalg.norm(np.diag(p) - np.outer(p, p)))
        rows[name] = np.mean(top), np.mean(grad)
    print(f"{d:4d}   {dots.std():10.2f}   {(dots / np.sqrt(d)).std():18.2f}   "
          f"{rows['raw'][0]:20.2f}   {rows['scaled'][0]:6.2f}   {rows['raw'][1]:6.3f}   {rows['scaled'][1]:6.3f}")
