# Semantic and Hybrid Search: the full code

The scripts behind [Semantic and Hybrid Search: BM25, Dense Retrieval and Rank Fusion](https://residual-stream.lvtan71.workers.dev/posts/semantic-and-hybrid-search/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `bm25_worked.py` | BM25 for the query "paged KV cache" on the blog's English sections, term by term: document frequency, idf, saturated count and contribution to the best section's score. | CPU, instant (numpy) |
| `retrieval.py` | The benchmark: each section heading is a query for its own section. How much each encoder cuts off; then BM25, two MiniLM encoders (whole sections and best 80-word passage) and reciprocal rank fusion on four tasks (English, Vietnamese, Vietnamese without diacritics, Vietnamese to English), with recall@1, recall@10, MRR, nDCG@10 and bootstrap intervals; and BM25 on a diacritic-stripped index. | CPU, about 2 minutes; downloads `all-MiniLM-L6-v2` (about 90 MB) and `paraphrase-multilingual-MiniLM-L12-v2` (about 470 MB) at pinned revisions |

Helpers: `sections.py` splits the posts in `corpus/en/` and `corpus/vi/` into sections (heading, text), dropping code, math, link targets and the headings every post shares; `bm25.py` is Okapi BM25 (k1 = 1.5, b = 0.75) and its tokenizer.

`blog-post37.tar.gz` links to `../../data/blog-post37.tar.gz`, a snapshot of the blog's English and Vietnamese posts (blog commit 2097c34, plus the post "Vector Search and ANN Indexes" at 4e96582). `tar xzf` unpacks it into `corpus/`. The post's run read a copy of the blog taken mid-session, with a draft of one later post that isn't in git; this snapshot has 308 English sections where the post had 309, so the last digits can differ. For example, `bm25_worked.py` on this snapshot finds "cache" in 69 sections, not 73, and a score of 10.26, not 10.20.

The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17, on the CPU.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.17.*" sentence-transformers
tar xzf blog-post37.tar.gz
python bm25_worked.py
python retrieval.py
```
