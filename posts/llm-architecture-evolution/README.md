# How LLM Architecture Evolved, 2017–2026: the full code

The scripts behind [How LLM Architecture Evolved, 2017–2026: A Short List of Substitutions](https://residual-stream.lvtan71.workers.dev/posts/llm-architecture-evolution/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `mlp_norm_swaps.py` | The tiny GPT with its SwiGLU MLP and RMSNorm, against a GELU MLP at 4D and against LayerNorm, one swap at a time, two seeds each, 4,000 steps on the dev slice. Runs `train.py` once per configuration. | GPU, about an hour (24,000 steps at about 0.15 s each on the M4) |
| `inference_memory.py` | The KV cache per generated token and the attention weights per layer for GPT-3 (MHA), Llama 2 70B and Llama 3 8B (GQA) and DeepSeek-V2 (MLA), and the parameters held against those used per token for two mixture-of-experts models. | Python, instant |
| `chinchilla.py` | Compute-optimal tokens per parameter from Chinchilla's scaling law as printed, with its unrounded constants, and as refit by Besiroglu et al.; then Llama 3 8B against the compute-optimal model at its training compute. | Python, instant |

Helpers: `gpt.py` and `train.py` are the tiny GPT and its training script from [Build a Tiny GPT, End to End, and Train It on This Blog](https://residual-stream.lvtan71.workers.dev/posts/tiny-gpt-from-scratch/). `train.py` is that post's file unchanged; `gpt.py` adds two options, `mlp` (`"swiglu"` or `"gelu"`, the GELU MLP 4D wide so the weights stay at 8D²) and `norm` (`"rms"` or `"layer"`). The post's runs used these files, which were not kept at the time; they are rebuilt here by replaying the recorded edits on the tiny GPT's files. `train.py` runs one configuration as `python train.py <name> '<json of run() arguments>'`.

`blog_train.txt` and `blog_val.txt` link to `../../data/`: the blog corpus the tiny GPT trained on, the English posts as of 2026-09-30 with two posts held out.

The run time of `mlp_norm_swaps.py` is an estimate from the step count. The outputs here came from an Apple M4 laptop with torch 2.14.0 and tokenizers 0.23.2. `mlp_norm_swaps.out` was retyped from the post's lines; it is byte-identical to the original run's output, which was not kept as a file. Seeds are fixed, but runs on another device or torch version can differ in the third decimal.

```bash
pip install "torch==2.14.*" "tokenizers==0.23.*"
python mlp_norm_swaps.py
python inference_memory.py
python chinchilla.py
```
