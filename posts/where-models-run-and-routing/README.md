# Where Models Run: the full code

The scripts behind [Where Models Run: Cloud, Device, and Routing Between Models](https://residual-stream.lvtan71.workers.dev/posts/where-models-run-and-routing/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `route.py` | Qwen2.5-1.5B-Instruct (small) and Qwen2.5-3B-Instruct (large) answer 150 checkable questions (capitals, two-digit multiplications, day-of-week sums); saves the answers to `answers.json`, and prints accuracy by kind, a confidence cascade at five thresholds, and the time per question. | GPU, about 2 minutes; downloads both models (3 GB and 6 GB) at pinned revisions. Time it with nothing else using the GPU |
| `rescore.py` | Rescores `answers.json` with accents folded and documented alternative capitals accepted, and compares the cascade with routing by question type, with bootstrap intervals. Rebuilds the questions by running the lines of `route.py` that make them. | CPU, instant, but the full install: the lines of `route.py` it runs import torch, transformers and huggingface_hub (no download). Reads `answers.json` from `route.py` |

The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17. The numpy and huggingface_hub versions were not recorded; numpy is pinned to its major version only.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.17.*" huggingface_hub
python route.py
python rescore.py
```
