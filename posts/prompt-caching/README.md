# Prompt Caching: the full code

The scripts behind [Prompt Caching: Paying Once for a Prefix](https://residual-stream.lvtan71.workers.dev/posts/prompt-caching/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `breakeven.py` | The break-even of a 5-minute and a 1-hour cache from the published price multipliers, the bill for a 20k-token prefix sent 1,000 times, and the KV cache a provider holds for that prefix on three models. | Python, instant |
| `ttft.py` | Time to the first token with and without a cached prefix, for prefixes of 1k to 8k tokens, on Qwen2.5-1.5B-Instruct. | GPU, about 3 minutes; downloads the model (3 GB) at a pinned revision. Reads `blog_train.txt`. Time it with nothing else using the GPU |

`lm.py` loads Qwen2.5-1.5B-Instruct and its tokenizer. `blog_train.txt` links to `../../data/blog-2026-10-04/blog_train.txt`: the blog's English posts as plain text, as they stood on 2026-10-04, written by the data script of [Sequences Before Attention](https://residual-stream.lvtan71.workers.dev/posts/rnns-and-why-transformers-won/).

The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.18.0.

```bash
pip install numpy "torch==2.14.*" "transformers==5.18.*" huggingface_hub
python breakeven.py
python ttft.py
```
