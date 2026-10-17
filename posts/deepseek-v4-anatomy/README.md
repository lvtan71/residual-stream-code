# Anatomy of a Frontier Model: the full code

The scripts behind [Anatomy of a Frontier Model: Reading DeepSeek-V4 Through Part V](https://residual-stream.lvtan71.workers.dev/posts/deepseek-v4-anatomy/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `kv_cache.py` | DeepSeek-V4-Pro's and V4-Flash's KV cache per token and at a million tokens, from the published configuration, against DeepSeek-V3.2 and a 61-layer GQA-8 model; and Pro's split between CSA and HCA layers. | Python, instant |

The outputs here came from an Apple M4 laptop; the script needs only Python.

```bash
python kv_cache.py
```
