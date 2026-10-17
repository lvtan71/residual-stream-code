import numpy as np

# Why trees fail in high dimensions: for 10,000 random points, the nearest
# and the farthest neighbor of a random query end up almost equally far away.
rng = np.random.default_rng(0)
for dim in (2, 10, 100, 384):
    P = rng.normal(size=(10_000, dim)); q = rng.normal(size=dim)
    dist = np.linalg.norm(P - q, axis=1)
    print(f"random points, {dim:3d} dimensions: nearest / farthest distance = {dist.min() / dist.max():.2f}")

# The real sentence embeddings of the next section (384 dimensions), for
# 200 of the query sentences: median of the same ratio.
X, Q = np.load("docs.npy"), np.load("queries.npy")
r = [np.sqrt(np.maximum(2 - 2 * (X @ q).max(), 0)) / np.sqrt(2 - 2 * (X @ q).min()) for q in Q[:200]]
print(f"real embeddings,   384 dimensions: nearest / farthest distance = {np.median(r):.2f} (median of 200 queries)")