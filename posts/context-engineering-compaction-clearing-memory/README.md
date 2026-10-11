# Context Engineering: the full code

The scripts behind [Context Engineering: Compaction, Tool Clearing, and Memory](https://residual-stream.lvtan71.workers.dev/posts/context-engineering-compaction-clearing-memory/). Each `.out` file is what its script printed for the post, and `results/` holds each run's measurements (context per turn, clearing and compaction events, file reads, summaries) for the figures.

| Script | What it does | Needs |
| --- | --- | --- |
| `harness.py` | The research agent: a synthetic corpus of six reviews with planted facts, three tools, an instrumented agent loop, a summary probe and a client-side memory backend. Imported by the others. | |
| `scratch.py` | Clearing as a pass over a message list, and the memory backend's path guard, without the API. | Python, instant, no key |
| `kvcost.py` | What the measured context sizes would hold in a KV cache for three open models. | Python, instant, no key |
| `baseline.py` | The agent with no context management, and what fills its window. | API key, about $0.40 |
| `clearing.py` | Tool-result clearing with keep = 2, 4 and 6, three runs of each setting that clears. | API key, about $3.50 |
| `compaction.py` | Compaction with the default prompt and with instructions, on the plain and the dense corpus, three runs each, each summary probed. | API key, about $3.50 |
| `memory.py` | Two sessions with the memory tool; the second run with and without the first one's notes. | API key, about $0.85 |
| `combined.py` | All three together. | API key, about $0.45 |

The runs used `claude-sonnet-5-5` with `anthropic` 1.13.0, on 2026-10-12. Each API script prints its token usage and cost at Sonnet 5.5's list prices. The model is not deterministic, so a rerun gives different turns, reads and summaries; the post reports three runs where one would mislead.

```bash
pip install "anthropic==1.13.*"
export ANTHROPIC_API_KEY=...
python scratch.py
python kvcost.py
python baseline.py
python clearing.py
python compaction.py
python memory.py
python combined.py
```
