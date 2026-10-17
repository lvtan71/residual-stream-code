# Beyond Vector RAG: the full code

The scripts behind [Beyond Vector RAG: Agentic, Graph and Vectorless Retrieval](https://residual-stream.lvtan71.workers.dev/posts/agentic-graph-and-vectorless-rag/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `graphrag_cost.py` | What a GraphRAG index of this blog would cost, against a vector index: the corpus in tokens and chunks, the LLM calls and tokens of indexing at the paper's ratios, and the tokens one global-search query reads. | CPU, under a minute; downloads Qwen2.5-1.5B-Instruct at a pinned revision (about 3 GB, only its tokenizer is used). Reads `content/posts/` from the blog snapshot |
| `compare.py` | 30 hand-written questions about the blog, each answered by one known section: BM25, dense (MiniLM), hybrid, and vectorless navigation by Qwen2.5-3B-Instruct in three variants and two modes, plus the reversed-list and section-only checks. | GPU, about 45 minutes; downloads all-MiniLM-L6-v2 (about 90 MB) and Qwen2.5-3B-Instruct (about 6 GB) at pinned revisions. Reads the blog snapshot |

`blog-b602ebd.tar.gz` links to `../../data/blog-b602ebd.tar.gz`: the blog's `content/` folder at commit `b602ebd`, the version both scripts read for the post. `tar xzf` unpacks it into `content/` here.

The whole folder runs in about 45 minutes; the per-script times are estimates from that total. The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.18.0. LLM picks on other hardware or library versions can differ by a question or two.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.18.*" huggingface_hub sentence-transformers
tar xzf blog-b602ebd.tar.gz
python graphrag_cost.py
python compare.py
```
