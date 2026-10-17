# Evaluating and Observing Agents: the full code

The scripts behind [Evaluating and Observing Agents: Outcomes, Trajectories, Traces and Spans](https://residual-stream.lvtan71.workers.dev/posts/evaluating-and-observing-agents/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `pass_k.py` | pass@k and pass^k for one task at several per-attempt success rates, and their unbiased estimates from n attempts with c successes. | CPU, instant; plain Python |
| `traced_agent.py` | A calculator agent on Qwen2.5-1.5B-Instruct: 20 two-step word problems, 5 runs each at temperature 0.7, every model and tool call recorded as a span with OpenTelemetry's GenAI attribute names. Prints one line per task and saves the traces to `traces.json`. Its printed lines were not kept, so it has no `.out`. | GPU (Apple M4: about 6 minutes); downloads Qwen2.5-1.5B-Instruct (about 3 GB) at a pinned revision |
| `evaluate_traces.py` | Reads `traces.json` and the answer key: pass@k and pass^k scored two ways, each run's trajectory, calculator use, a dashboard of calls, tokens and latency, and one trace printed as a waterfall. | CPU, instant (numpy) |

The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17. Generation is sampled with fixed seeds, but sampling on a different device or library version can choose different tokens, so a rerun's counts and timings can differ.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.17.*" huggingface_hub
python pass_k.py
python traced_agent.py
python evaluate_traces.py
```
