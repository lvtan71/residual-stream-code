# Evaluating LLMs: the full code

The scripts behind [Evaluating LLMs: Metrics, Benchmarks, Judges and Error Bars](https://residual-stream.lvtan71.workers.dev/posts/evaluating-llms/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `sample_size.py` | How many questions it takes to detect 70% against 72%, unpaired and paired; Wilson intervals for one accuracy; the interval on a difference at 500 questions; Cohen's kappa for two raters. | Python, instant |
| `judge.py` | Qwen2.5-1.5B-Instruct as a pairwise judge on 100 multiplications and 20 capitals with known answers: position bias, and how claims of checking or neutral padding change its choice. | GPU, a few minutes (CPU works, slower); downloads Qwen2.5-1.5B-Instruct (about 3 GB) at a pinned revision |
| `arena.py` | Bradley–Terry ratings fitted to simulated arena votes for four models with known ratings, at 500, 5,000 and 50,000 votes, with 95% bootstrap intervals. | numpy, under a minute |

The whole folder runs in about 5 minutes; the per-script times are estimates from that total. The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17. The judge runs in float16 on whichever device is available; on another GPU a judgment whose two letters are nearly tied could flip.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.17.*" huggingface_hub
python sample_size.py
python judge.py
python arena.py
```
