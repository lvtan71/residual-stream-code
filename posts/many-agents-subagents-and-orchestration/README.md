# Many Agents: the full code

The scripts behind [Many Agents: Subagents, Communication and Orchestration](https://residual-stream.lvtan71.workers.dev/posts/many-agents-subagents-and-orchestration/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `cost_model.py` | Counts the tokens of the six posts one research question needs, then a cost model of one agent reading them against a lead with six workers, under three sets of assumptions: tokens read, cost with prompt caching, largest context and sequential calls. | CPU, a few seconds; downloads the Qwen2.5-1.5B-Instruct repository at a pinned revision (about 3 GB; only its tokenizer is used). Reads `content/posts/` from the blog snapshot |

`blog-095741d.tar.gz` links to `../../data/blog-095741d.tar.gz`: the blog's source at commit 095741d, the posts as they were when this post was written. `tar xzf` unpacks it into `content/` and `images/`.

The whole folder runs in about a minute on a CPU. The outputs here came from an Apple M4 laptop with transformers 5.17 (the tokenizer only; torch is not needed).

```bash
pip install "transformers==5.17.*" huggingface_hub
tar xzf blog-095741d.tar.gz
python cost_model.py
```
