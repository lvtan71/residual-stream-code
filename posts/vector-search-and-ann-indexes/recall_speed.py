import time
import faiss
import numpy as np

# Exact and approximate nearest-neighbor search over 71,865 real sentence
# embeddings (384 dimensions, normalized), 1,000 held-out queries, top 10.
# Truth: exact inner-product search. Every index is queried one query at a
# time on one CPU thread, the latency a single request sees; each setting
# is timed three times and the fastest kept, since other work on the
# machine only ever slows a run down.
faiss.omp_set_num_threads(1)
X, Q = np.load("docs.npy"), np.load("queries.npy")
N, d, k = *X.shape, 10
exact = faiss.IndexFlatIP(d); exact.add(X)
_, truth = exact.search(Q, k)

# On unit vectors, L2, dot product and cosine rank identically:
#    |q - x|^2 = 2 - 2 q.x.
l2 = faiss.IndexFlatL2(d); l2.add(X); _, by_l2 = l2.search(Q[:100], k)
same = all(set(a) == set(b) for a, b in zip(by_l2, truth[:100]))
swaps = [(i, j) for i in range(100) for j in range(k) if by_l2[i, j] != truth[i, j]]
ties = all(np.isclose(Q[i] @ X[by_l2[i, j]], Q[i] @ X[truth[i, j]]) for i, j in swaps)
print(f"L2 vs dot product, 100 queries: same top-10 sets: {same}; order differs at {len(swaps)} positions, "
      f"all between exactly tied scores (duplicate sentences): {ties}")

def measure(index, name, bytes_per_vector, setter=None, values=(None,)):
    for v in values:
        if setter:
            setter(v)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            found = np.vstack([index.search(Q[i:i + 1], k)[1] for i in range(len(Q))])
            times.append(time.perf_counter() - t0)
        ms = min(times) / len(Q) * 1e3
        recall = np.mean([len(set(f) & set(t)) / k for f, t in zip(found, truth)])
        label = name if v is None else f"{name}, {v}"
        print(f"{label:32s} recall@10 {recall:6.3f}   {ms:6.3f} ms/query   {bytes_per_vector:5.0f} bytes/vector")

print(f"\n{N:,} vectors, {d} dimensions:")
measure(exact, "flat (exact)", 4 * d)
t0 = time.perf_counter()
ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 256, faiss.METRIC_INNER_PRODUCT); ivf.train(X); ivf.add(X)
print(f"  (IVF, 256 lists: built in {time.perf_counter() - t0:.1f} s)")
measure(ivf, "IVF 256 lists, probe", 4 * d + 8, lambda v: setattr(ivf, "nprobe", v), (1, 4, 16, 64))
# FAISS's guideline is 4 to 16 times sqrt(N) lists; sqrt(71,865) is 268.
ivf4 = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 1024, faiss.METRIC_INNER_PRODUCT); ivf4.train(X); ivf4.add(X)
measure(ivf4, "IVF 1024 lists, probe", 4 * d + 8, lambda v: setattr(ivf4, "nprobe", v), (8, 16, 32, 64))
t0 = time.perf_counter()
pq = faiss.IndexPQ(d, 48, 8, faiss.METRIC_INNER_PRODUCT); pq.train(X); pq.add(X)
print(f"  (PQ, 48 sub-vectors x 8 bits: built in {time.perf_counter() - t0:.1f} s)")
measure(pq, "PQ, 48-byte codes", 48)
ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, 256, 48, 8, faiss.METRIC_INNER_PRODUCT); ivfpq.train(X); ivfpq.add(X)
measure(ivfpq, "IVF-PQ, probe", 48 + 8, lambda v: setattr(ivfpq, "nprobe", v), (4, 16, 64))
# PQ's recall is capped by its coarse codes; re-scoring the top 100
# candidates with the exact vectors (kept in memory, or on disk) lifts it.
base = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, 256, 48, 8, faiss.METRIC_INNER_PRODUCT); base.train(X)
refine = faiss.IndexRefineFlat(base); refine.k_factor = 10; refine.add(X)
measure(refine, "IVF-PQ + re-rank 100, probe", 48 + 8 + 4 * d, lambda v: setattr(base, "nprobe", v), (16, 64))
t0 = time.perf_counter()
hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); hnsw.hnsw.efConstruction = 128; hnsw.add(X)
print(f"  (HNSW, M = 32: built in {time.perf_counter() - t0:.1f} s)")
measure(hnsw, "HNSW M=32, efSearch", 4 * d + 2 * 32 * 4, lambda v: setattr(hnsw.hnsw, "efSearch", v), (16, 32, 64, 128))
lsh = faiss.IndexLSH(d, 1024); lsh.add(X)
measure(lsh, "LSH, 1024-bit codes, scanned", 1024 / 8)