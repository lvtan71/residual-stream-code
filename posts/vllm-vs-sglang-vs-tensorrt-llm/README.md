# vLLM vs. SGLang vs. TensorRT-LLM: the full code

The script behind [vLLM vs. SGLang vs. TensorRT-LLM: Comparing LLM Serving Architectures](https://residual-stream.lvtan71.workers.dev/posts/vllm-vs-sglang-vs-tensorrt-llm/). `reuse.out` is what it printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `radix.py` | Two prefix caches with one interface: a radix tree matching any prefix to the token (SGLang's), and hashed blocks of any size (vLLM's 16 tokens, TensorRT-LLM's 32). The same file as in the SGLang post's folder. | |
| `reuse.py` | Replays multi-turn ShareGPT chat through each cache at three cache sizes and five match granularities. | CPU, under a minute; downloads ShareGPT (673 MB) and Qwen2.5's tokenizer from Hugging Face at pinned revisions |

```bash
pip install transformers huggingface_hub
python reuse.py
```
