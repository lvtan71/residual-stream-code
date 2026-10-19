# Agent Architectures: the full code

The scripts behind [Agent Architectures: ReAct, Plan-and-Execute and Reflection, Compared](https://residual-stream.lvtan71.workers.dev/posts/agent-architectures-react-plan-reflection/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `arch.py` | Twelve fictional companies, `lookup` and `calculator` tools, and 20 questions; ReAct, plan-and-execute and reflection run each question three times with Qwen2.5-3B-Instruct at temperature 0.7. Prints accuracy, tokens and seconds per question, and logs every trial to `arch_trials_v3.jsonl` (a stopped run resumes from it). | GPU, about 87 minutes on the M4; downloads Qwen2.5-3B-Instruct (about 6 GB) at a pinned revision. Time it with nothing else using the GPU |
| `failures.py` | Rebuilds the same questions and sorts every logged trial: right, sign flipped, a total missing one company, no answer, other; accuracy by question type; paired comparisons; and what reflection's critic flagged and what its retries changed; then the counts the post quotes from the replies (sign flips, failed lookups the reply mentions, what the critic said, unanswered retries). Reads `arch_trials_v3.jsonl`, or the log named as its argument. | CPU, instant |

The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.18.0 (the corrected rerun of 2026-10-04). Sampling is seeded per trial, but generation on another device or library version can take a different path, so the counts can differ. A second run on the same M4 and versions on 2026-10-19 reproduced every answer count and token count in `arch.out`; only the seconds differed. Its trial log is committed here as `arch_trials_m4.jsonl`: `python failures.py arch_trials_m4.jsonl` reprints `failures.out` without running `arch.py`.

```bash
pip install "torch==2.14.*" "transformers==5.18.*" huggingface_hub numpy
python arch.py
python failures.py
```
