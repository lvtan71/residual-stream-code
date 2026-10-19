# Mixture of Experts: the full code

The scripts behind [Mixture of Experts: More Parameters, Same FLOPs](https://residual-stream.lvtan71.workers.dev/posts/mixture-of-experts/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `balancing.py` | The tiny GPT with 8 experts in place of each MLP, on the blog's characters: top-1 with the balancing loss, with no balancing and with bias balancing, and top-2 with the balancing loss, two seeds each, 4,000 steps on the dev slice. Prints bits per byte at the checkpoints and the busiest expert's share in each layer. | GPU, roughly 2 to 3 hours (32,000 steps; an MoE step is slower than a dense one) |
| `overfit.py` | The dense model and the top-1 MoE again, seed 0, on the CPU, also scoring 60,000 characters of the training text at each checkpoint. | CPU, about an hour |
| `wikitext2.py` | Downloads WikiText-2 (raw) at a pinned revision and writes `wikitext2_train.txt` and `wikitext2_test.txt`. | Downloads about 7 MB; needs pyarrow |
| `wikitext2_runs.py` | The dense model and the MoE, top-1 and top-2, on WikiText-2, two seeds each, 4,000 steps. Reads the files `wikitext2.py` writes. | GPU, roughly 2 to 3 hours (24,000 steps) |
| `moe_arithmetic.py` | Mixtral 8x7B's and DeepSeek-V3's expert weights, total and used per token; the all-to-all traffic per token under expert parallelism; and how many experts a decode batch touches. | Python, instant |

Helpers: `moe.py` is the mixture-of-experts layer (router, experts, and the three balancing methods). `gpt.py` and `train.py` are the tiny GPT and its training script from [Build a Tiny GPT, End to End, and Train It on This Blog](https://residual-stream.lvtan71.workers.dev/posts/tiny-gpt-from-scratch/), extended: `gpt.py` has the `mlp` and `norm` options of [How LLM Architecture Evolved, 2017–2026](https://residual-stream.lvtan71.workers.dev/posts/llm-architecture-evolution/) and this post's `experts`, `top_k`, `expert_width`, `balance` and `renorm_top1` (the broken top-1 gate); `train.py` adds the balancing loss and the busiest-expert share, puts the experts' stacked 3-D weights into the weight-decay group (the fix for the bug the post describes), and has two run options, `track_fit` (also score training text) and `corpus` (`"wikitext2"` trains on WikiText-2, scoring on 120,000 characters of its held-back 10%). The post's runs used these files, which were not kept at the time; they are rebuilt here by replaying the recorded edits on the tiny GPT's files. `train.py` runs one configuration as `python train.py <name> '<json of run() arguments>'`. The one-expert check in the post's bug section has no script of its own: after the fix it is `python train.py "1 expert" '{"kind": "char", "steps": 1000, "dropout": 0.2, "seed": 0, "experts": 1, "top_k": 1, "balance": "none", "renorm_top1": true}'`, which printed 2.138 at step 1,000 (the post's 2.14); the 2.73 before the fix needs `train.py`'s optimizer line changed back to `p.dim() == 2`.

`blog_train.txt` and `blog_val.txt` link to `../../data/`: the blog corpus the tiny GPT trained on, the English posts as of 2026-09-30 with two posts held out.

Run times are estimates from the step counts. The outputs here came from an Apple M4 laptop with torch 2.14.0 and tokenizers 0.23.2. `balancing.out`, `overfit.out`, `wikitext2.out` and `wikitext2_runs.out` were retyped from the post; the original files were not kept. The notebook's rerun reproduces `overfit.out` (on the M4) and `wikitext2.out` exactly; its GPU runs, on an RTX 3090, differ from `balancing.out` and `wikitext2_runs.out` by up to 0.04 bits per byte, and the busiest-expert shares by up to 4 points (13 in the runs without balancing). Seeds are fixed, but runs on another device or torch version can differ by a few hundredths.

```bash
pip install "torch==2.14.*" "tokenizers==0.23.*" huggingface_hub pyarrow
python balancing.py
python overfit.py
python wikitext2.py
python wikitext2_runs.py
python moe_arithmetic.py
```
