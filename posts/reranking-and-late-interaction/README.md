# Reranking and Late Interaction: the full code

The scripts behind [Reranking and Late Interaction: Cross-Encoders to ColBERT](https://residual-stream.lvtan71.workers.dev/posts/reranking-and-late-interaction/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `cost.py` | Cost per query of a bi-encoder, a cross-encoder over the top 50, and ColBERT's late interaction, for the blog's English sections; the same at 10,000 and 1,000,000 documents; and each design's index size. | CPU, under a minute; downloads the all-MiniLM-L6-v2 tokenizer at a pinned revision |
| `maxsim.py` | ColBERT's MaxSim worked on a three-word query against two short documents, with all-MiniLM-L6-v2's token vectors, against a mean of similarities. | CPU, a few seconds; downloads all-MiniLM-L6-v2 (about 90 MB) at a pinned revision |
| `rerank.py` | Two-stage retrieval on the blog's sections: the best first stage from Semantic and Hybrid Search, then two cross-encoders over its top 50, in English and Vietnamese, with paired MRR differences and the time per query. | CPU, about 30 minutes; downloads two embedding models and two cross-encoders (under 1 GB in all) at pinned revisions |

`blog-post37.tar.gz` links to `../../data/blog-post37.tar.gz`, a snapshot of the blog's posts (`corpus/en`, `corpus/vi`) rebuilt from git: commit `2097c34` plus `vector-search-and-ann-indexes.mdx` from `4e96582`. The post's run read a copy of the blog taken mid-session, with a draft post that isn't in git, so this snapshot has 308 English sections where the post had 309, and the last digits can differ.

`sections.py` (the benchmark: each heading is a query, its section the one relevant document) and `retrieval.py` (BM25, the dense encoders, fusion and scoring) are copies of the helpers from [Semantic and Hybrid Search](https://residual-stream.lvtan71.workers.dev/posts/semantic-and-hybrid-search/). `retrieval.py` here stops before that post's encoder measurements, so importing it only loads the data and the retrievers.

The whole folder runs in about 35 minutes on a CPU; the per-script times are estimates from that total. The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.17.*" sentence-transformers
tar xzf blog-post37.tar.gz
python cost.py
python maxsim.py
python rerank.py
```
