# Speculative Decoding, Derived: the full code

The scripts behind [Speculative Decoding, Derived: Draft Models, N-grams, Medusa and EAGLE](https://residual-stream.lvtan71.workers.dev/posts/speculative-decoding/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `spec_math.py` | Checks by sampling that the acceptance rule reproduces the target's distribution with a bad draft; tokens per verification step, formula against simulation; speedup by draft cost. | numpy, a few seconds |
| `accept.py` | Qwen2.5-1.5B-Instruct as the target, Qwen2.5-0.5B as the draft: per-position greedy agreement, sum of min(p, q), and two-token prompt lookup, on paragraph rewrites and open questions. | GPU or CPU, about 4 minutes (estimate); downloads both models at pinned revisions, about 4 GB. Reads `blog_val.txt` |
| `spec_steps.py` | The same models and prompts, measured as speculation runs: tokens per step for greedy speculation at k = 4 and 8 against the formula, acceptance on sampled text, and prompt lookup as deployed. | GPU or CPU, about 6 minutes (estimate); same models. Reads `blog_val.txt` |

`blog_val.txt` links to `../../data/blog_val.txt`: the blog corpus that [Sequences Before Attention](https://residual-stream.lvtan71.workers.dev/posts/rnns-and-why-transformers-won/) froze, the English posts as of 2026-09-30, validation split.

`spec_steps.out` holds the lines the post printed; the full output was not kept. The script also prints the two vocabulary sizes, one line per prompt, and a `prompt lookup pl2` line per task (lookup limited to two-token matches).

The whole folder runs in about 10 minutes; the per-script times are estimates from that total. The outputs here came from an Apple M4 laptop, on its CPU, with torch 2.14.0 and transformers 5.17. Both model scripts run on the CPU, as the post's run did. On a CUDA or Apple GPU the greedy answers, and so every count, can change, and the sampled run will differ.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.17.*"
python spec_math.py
python accept.py
python spec_steps.py
```
