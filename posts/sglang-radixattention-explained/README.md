# SGLang and RadixAttention: the full code

The complete scripts behind [SGLang and RadixAttention: Sharing the KV Cache Across Requests](https://residual-stream.lvtan71.workers.dev/posts/sglang-radixattention-explained/). The post shows the important parts of each script, and each `.out` file is what the script printed for the post.

Run them from this folder:

| Script | What it does | Needs |
| --- | --- | --- |
| `radix.py` | Two prefix caches with one interface: SGLang's radix tree (any prefix, to the token; LRU over leaves) and vLLM's hashed 16-token blocks. Imported by the next two. | |
| `hits.py` | Reuse on multi-turn ShareGPT conversations, where each turn's prompt is the whole conversation so far, at five cache sizes. | CPU, about a minute; downloads ShareGPT (673 MB) from Hugging Face at a pinned revision |
| `schedule.py` | Cache-aware scheduling on SGLang's shared-prefix benchmark design: arrival order, longest-prefix-match first, and longest-prefix-match as SGLang ships it. | CPU, under a minute. Reads `blog_train.txt`, a link to `../../data/blog_train.txt` |
| `prefill.py` | Time to the first token of a 2,176-token prompt, computed whole and with its 2,048-token prefix cached. | GPU; time it with nothing else using the GPU |
| `jump.py` | Constrained JSON decoding, plain and with jump-forward: forward passes and time. | GPU |

`lm.py` loads Qwen2.5-1.5B-Instruct and its tokenizer; the other scripts import it.

```bash
pip install torch transformers huggingface_hub
python hits.py
```
