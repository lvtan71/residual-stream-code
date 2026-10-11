# vLLM Architecture: the full code

The complete scripts behind [vLLM Architecture: Continuous Batching and Serving at Scale](https://residual-stream.lvtan71.workers.dev/posts/vllm-architecture-explained/). The post shows the important parts of each script, and each `.out` file is what the script printed for the post.

Run them from this folder:

| Script | What it does | Needs |
| --- | --- | --- |
| `steps.py` | Times decode steps (by batch size and cached tokens) and prompt prefills of Qwen2.5-1.5B, then fits the cost model the simulation uses. Writes `cost.npy` (included here). | GPU; about 5 minutes on the post's Apple M4; time it with nothing else using the GPU. Reads `blog_train.txt`, a link to `../../data/blog_train.txt` |
| `engine.py` | The scheduler simulation: static batching, prefill-first and V1-style chunked continuous batching, with a block pool and preemption, on ShareGPT request lengths. | CPU, seconds. Reads `cost.npy` and `sharegpt_lengths.json` (written by `waste.py` in [the paged attention folder](../paged-attention-explained/)) |
| `lora.py` | The cost of serving many LoRA adapters from one model: shared, gathered, and mixed. | GPU |
| `tp.py` | Tensor parallelism's all-reduce traffic for Llama-3-70B on eight H100s. | Nothing; arithmetic only |

`lm.py` loads the model and tokenizer; the other scripts import it. A different GPU will give different times, so `engine.py` run with your own `cost.npy` will give different numbers; the comparison between schedulers is the point.

```bash
pip install torch transformers huggingface_hub numpy
python engine.py
```
