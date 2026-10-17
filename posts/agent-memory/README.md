# Agent Memory: the full code

The scripts behind [Agent Memory: Write, Read, Update, Forget](https://residual-stream.lvtan71.workers.dev/posts/agent-memory/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `retrieval.py` | 250 synthetic 50-turn conversations (5 seeds × 50) with 8 facts about the user, 3 of them changed later. For each retrieval policy (last k turns, most relevant k, and the recency + importance + relevance score with two ablations) and budget k, the share of questions whose shown memories hold the current version only, both, the outdated version only, or neither; then one worked example's three scores. | CPU, about a minute; downloads the sentence encoder `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB) at a pinned revision |

The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17, on its CPU.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.17.*" "sentence-transformers==6.1.*"
python retrieval.py
```
