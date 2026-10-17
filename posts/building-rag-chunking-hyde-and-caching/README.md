# Building RAG: the full code

The scripts behind [Building RAG: Chunking, HyDE and Caching](https://residual-stream.lvtan71.workers.dev/posts/building-rag-chunking-hyde-and-caching/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `make_questions.py` | The test set: 150 sentences drawn from the blog's posts, and for each, Qwen2.5-3B-Instruct writes a question it answers, two rewordings and a hypothetical answer. Writes `gen_new.json`, so that it doesn't overwrite the linked `gen.json`. | GPU, about 30 minutes (600 generations); downloads Qwen2.5-3B-Instruct (about 6 GB) at a pinned revision |
| `retrieval.py` | Reads `gen.json`. Nine ways of chunking the posts, scored by dense retrieval with MiniLM (hits in the top 1 and top 5, words returned, sign tests); HyDE in four forms; an embedding cache; and a semantic cache at six thresholds, on rewordings, same-post pairs and twelve hand-written near-miss pairs. | CPU, a few minutes; downloads all-MiniLM-L6-v2 (about 90 MB) at a pinned revision |

`common.py` is the helper both scripts import: it reads the posts from `content/posts/` and turns each into clean text, leaving out the posts that discuss this post's experiments.

`blog-1e98683.tar.gz` links to `../../data/`: a snapshot of the blog's source at commit 1e98683; `tar xzf` unpacks its `content/posts/`. The post's run read the working tree at the time, which also held a draft of one more post, so it saw 67 posts where the snapshot gives 66; chunk counts and hits from the snapshot differ from `retrieval.out` by a few.

`gen.json` links to `../../data/gen.json`: the 150 questions the post used, as `make_questions.py` wrote them. `retrieval.py` reads it. A new run of `make_questions.py` should give the same questions (decoding is greedy), but half-precision arithmetic can differ between GPUs; to score a new set, rename `gen_new.json` to `gen.json`.

The run times are estimates from the 35 minutes the whole post took on the M4. The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.18.0.

```bash
pip install "torch==2.14.*" "transformers==5.18.*" huggingface_hub sentence-transformers numpy
tar xzf blog-1e98683.tar.gz
python make_questions.py
python retrieval.py
```
