# KV Cache Compression: the full code

The scripts behind [KV Cache Compression: Eviction and Quantization](https://residual-stream.lvtan71.workers.dev/posts/kv-cache-compression/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `budget.py` | The KV cache's bytes per token and size for one 32k-token sequence, for Llama-2-13B, under fewer KV heads, 4- and 2-bit quantization, keeping a quarter of the tokens, and all three combined. | Python, instant |
| `measure.py` | Qwen2.5-0.5B's loss in bits per token on the last 1,024 tokens of eight 2,048-token windows of held-out blog text, with the full cache, with eviction (recent windows, with and without 4 sink tokens), and with the cached keys and values quantized (8/4/2 bits, per token or per channel, KIVI's full-precision recent window), a control with the key bias removed before rounding, and the size of the key bias in two layers. | CPU, about 6 minutes after the download; downloads `Qwen/Qwen2.5-0.5B` (about 1 GB) at a pinned revision. Reads `blog_val.txt` |

`blog_val.txt` links to `../../data/blog_val.txt`, the held-out split of the blog's posts. `measure.py` runs the model on the CPU in float32 on every machine, as it did for the post; it uses no GPU.

The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17, on its CPU.

```bash
pip install "torch==2.14.*" "transformers==5.17.*" huggingface_hub
python budget.py
python measure.py
```
