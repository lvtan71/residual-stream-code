# Guardrails and Prompt Injection: the full code

The scripts behind [Guardrails and Prompt Injection: Why Filters Reduce Risk and Only Architecture Bounds It](https://residual-stream.lvtan71.workers.dev/posts/guardrails-and-prompt-injection/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `injection.py` | Indirect prompt injection, measured for one model size (the `SIZE` environment variable, `1.5B` or `3B`): Qwen2.5-Instruct summarizes 12 passages of the blog, each with 8 planted attacks and once clean, under five defenses (none, a "documents are data" rule, spotlighting, an input classifier, no email tool). Prints attack success with a 95% interval, every `send_email` call and what the classifier blocks; saves the replies to `samples_<size>.json`. | GPU; downloads the model at a pinned revision (1.5B about 3 GB, 3B about 6 GB) |
| `injection_both.py` | Runs `injection.py` with `SIZE=1.5B`, then `SIZE=3B`. | GPU; run time not recorded (5 × 108 generations of up to 200 tokens per size) |

`blog-77f8212.tar.gz` links to `../../data/blog-77f8212.tar.gz`, the blog's posts at commit `77f8212` (`content/posts`), from which the twelve passages are sampled with a fixed seed.

The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17. Decoding is greedy, but 16-bit arithmetic on another GPU can still change a few replies.

```bash
pip install "torch==2.14.*" "transformers==5.17.*" huggingface_hub
tar xzf blog-77f8212.tar.gz
python injection_both.py
```
