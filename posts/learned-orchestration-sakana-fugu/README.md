# Learned Orchestration: the full code

The script behind [Learned Orchestration: A Case Study of Sakana Fugu](https://residual-stream.lvtan71.workers.dev/posts/learned-orchestration-sakana-fugu/). Its `.out` file is what it printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `fugu_margins.py` | Table 1 of the Sakana Fugu technical report (arXiv 2606.21228v2), as reported: each orchestrator's margin over the best of its three workers on each of 11 benchmarks, how often each is ahead, and which worker is best where. | CPU, instant; plain Python, no packages |

The numbers are the report's, typed in from its Table 1; the script only does arithmetic on them. The output here came from an Apple M4 laptop with Python 3.9.6 (the system `python3`).

```bash
python fugu_margins.py
```
