# Engineering Around the Model: the full code

The scripts behind [Engineering Around the Model: Harnesses, Loops, Graphs and the Definition of Done](https://residual-stream.lvtan71.workers.dev/posts/engineering-around-the-model/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `done_check.py` | Two harnesses around Qwen2.5-1.5B-Instruct and a calculator tool on the 20 warehouse problems from Evaluating and Observing Agents, 3 runs each: the plain one stops at the first reply without a tool call; the checked one also requires the final number to equal the calculator's last result. Prints accuracy at step budgets of 2 to 8 model calls and the paired comparison, and saves every transcript to `runs.json`. | GPU, about 10 minutes; downloads the model (about 3 GB) at a pinned revision |

The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17. `runs.json` holds every run's transcript, which the post quotes; it was regenerated on 2026-10-19 by running `done_check.py` again on the same laptop and library versions, which printed `done_check.out` exactly, so it is the same set of runs. Sampling on another GPU can give different transcripts, and so different counts.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.17.*" huggingface_hub
python done_check.py
```
