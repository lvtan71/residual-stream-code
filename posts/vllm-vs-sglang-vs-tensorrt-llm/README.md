# vLLM vs. SGLang vs. TensorRT-LLM: the full code

The script behind [vLLM vs. SGLang vs. TensorRT-LLM: Comparing LLM Serving Architectures](https://residual-stream.lvtan71.workers.dev/posts/vllm-vs-sglang-vs-tensorrt-llm/). `reuse.out` is what `reuse.py` printed for the post. The output here came from an Apple M4 laptop with transformers 5.19.0.

| Script | What it does | Needs |
| --- | --- | --- |
| `radix.py` | Two prefix caches with one interface: a radix tree matching any prefix to the token (SGLang's), and hashed blocks of any size (vLLM's 16 tokens, TensorRT-LLM's 32). The same file as in the SGLang post's folder. | |
| `partial.py` | TensorRT-LLM's default: 32-token blocks with partial reuse, so a match can end inside a block. Imported by `reuse.py`. | |
| `reuse.py` | Replays multi-turn ShareGPT chat through each engine's default cache at three cache sizes, then through whole blocks of four sizes. | CPU, under a minute; downloads ShareGPT (673 MB) and Qwen2.5's tokenizer from Hugging Face at pinned revisions |

```bash
pip install "transformers==5.19.*" huggingface_hub
python reuse.py
```
