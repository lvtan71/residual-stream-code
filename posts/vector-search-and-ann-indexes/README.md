# Vector Search: the full code

The scripts behind [Vector Search: Exact Search and the Approximate Indexes](https://residual-stream.lvtan71.workers.dev/posts/vector-search-and-ann-indexes/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `embed.py` | Embeds every sentence of WikiText-2's training set over 40 characters (71,865) and 1,000 sentences of its test set with all-MiniLM-L6-v2, normalized, and saves `docs.npy` and `queries.npy`. Its printed line was not kept, so it has no `.out`. | CPU, several minutes; downloads WikiText-2 (a few MB) and all-MiniLM-L6-v2 (about 90 MB) at pinned revisions |
| `distances.py` | The ratio of the nearest to the farthest distance from a query, for random points in 2 to 384 dimensions and for the real sentence embeddings. Reads `docs.npy`, `queries.npy`. | CPU, a few seconds |
| `recall_speed.py` | Recall@10, time per query and bytes per vector for exact search, IVF (256 and 1,024 lists), PQ, IVF-PQ with and without re-ranking, HNSW and random-hyperplane codes, one query at a time on one thread. Reads `docs.npy`, `queries.npy`. | CPU, about 2 minutes (HNSW's build alone is 47 s). Time it with nothing else running |

The whole folder runs in about 6 minutes on a CPU; the per-script times are estimates from that total and from the build times printed. The outputs here came from an Apple M4 laptop with torch 2.14.0, transformers 5.17.0, sentence-transformers 6.1.0 and faiss-cpu 1.15.1, except `recall_speed.out`, rerun on 2026-10-18 on the same laptop with nothing else running (torch 2.14.1); two such runs agreed within about 4%. The embeddings are computed on the CPU; times per query depend on the machine.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.17.*" "sentence-transformers==6.1.*" "faiss-cpu==1.15.*" pyarrow huggingface_hub
python embed.py
python distances.py
python recall_speed.py
```
