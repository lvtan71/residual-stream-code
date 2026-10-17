# Reasoning Models and Test-Time Compute: the full code

The scripts behind [Reasoning Models and Test-Time Compute](https://residual-stream.lvtan71.workers.dev/posts/reasoning-and-test-time-compute/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `cost.py` | What thinking tokens cost on one H100 for Qwen2.5-32B and DeepSeek-R1: KV cache size and bandwidth-bound decode time at 1,000, 10,000 and 50,000 tokens. | CPU, instant |
| `lm.py` | Loads Qwen2.5-1.5B-Instruct and its tokenizer, at a pinned revision, on the first call. | helper, no output; downloads the model (about 3 GB) |
| `tts.py` | 100 three-step word problems answered directly, with a greedy chain, and with 16 sampled chains combined by majority vote; one seed per call (`python tts.py 0`). | GPU, about 43 minutes for seed 0 and 30 for seed 1 on the M4 |
| `tts_seeds.py` | Runs `tts.py` with seeds 0 and 1, as the post did. | as `tts.py`, about 73 minutes |
| `chain.py` | A copy of the experiment script of [Chain-of-Thought and Prompt Chaining](https://residual-stream.lvtan71.workers.dev/posts/chain-of-thought-and-prompt-chaining/). Only its problem generator is used here, by `tts_hard.py`. | not run in this folder |
| `tts_hard.py` | 40 problems of eight dependent operations, 16 sampled chains each: majority of k, some sample right, and one line per problem. | GPU, about 13 minutes (estimate) |
| `hard_stats.py` | Average one-sample accuracy and unbiased pass@k from the per-problem counts in `tts_hard.out`. | CPU, instant |

`hard_stats.py` reads `tts_hard.out`, the post's saved log of `tts_hard.py`, which is committed here. To use a new run instead, save its output there first: `python tts_hard.py > tts_hard.out`.

The whole folder runs in about 86 minutes on the GPU of an Apple M4 laptop; the per-script times are from the printed timings, and `tts_hard.py`'s is the remainder. The outputs here came from that laptop with torch 2.14.1 and transformers 5.18.0. Sampled runs on other hardware or library versions can give slightly different counts.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.18.*" huggingface_hub
python cost.py
python tts_seeds.py
python tts_hard.py
python hard_stats.py   # reads tts_hard.out, the post's saved log of tts_hard.py
```
