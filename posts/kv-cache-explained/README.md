# The KV Cache: the full code

The scripts behind [The KV Cache: How Transformers Avoid Recomputing the Past](https://residual-stream.lvtan71.workers.dev/posts/kv-cache-explained/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `toy.py` | The post's 4-token example, decoded with and without a cache. | numpy, instant |
| `budget.py` | The cache's memory formula for several models, and an idealized bandwidth-bound decode on one H100 for Llama 2 13B. | Python, instant |
| `real.py` | Qwen2.5-1.5B-Instruct: cached and recomputed logits compared, the cache's measured size, one step's time with and without the cache, and decode time by batch size. | GPU, about five minutes; downloads the model (3 GB) at a pinned revision. Reads `blog_train.txt`, a link to `../../data/blog_train.txt`. Time it with nothing else using the GPU |

`lm.py` loads Qwen2.5-1.5B-Instruct and its tokenizer. The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.19.

```bash
pip install numpy "torch==2.14.*" "transformers==5.19.*" huggingface_hub
python toy.py
python budget.py
python real.py
```
