# System Design for AI: the full code

The scripts behind [System Design for AI: A Real-Time Voice Agent, Millisecond by Millisecond](https://residual-stream.lvtan71.workers.dev/posts/voice-agent-system-design/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `latency_budget.py` | A cascaded voice agent's response time, stage by stage, from assumed figures, and the effect of two levers: semantic turn detection and a speech-to-speech model. | Python, instant |
| `capacity.py` | An idealized simulation of continuous batching for the LLM stage of 1,000 calls on H100s: time to first token and between tokens for 1 to 12 GPUs, with each call's history kept in GPU memory, recomputed every turn, or copied back from CPU memory. | numpy, a few seconds |

The whole folder runs in under a minute on a CPU. Neither script measures hardware: both compute from the assumptions stated at their top. The outputs here came from an Apple M4 laptop; `capacity.py` needs only numpy.

```bash
pip install "numpy==2.*"
python latency_budget.py
python capacity.py
```
