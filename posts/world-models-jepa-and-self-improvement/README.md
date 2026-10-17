# World Models, JEPA and Recursive Self-Improvement: the full code

The scripts behind [World Models, JEPA and Recursive Self-Improvement: Three Bets Beyond the Next Token](https://residual-stream.lvtan71.workers.dev/posts/world-models-jepa-and-self-improvement/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `pendulum_imagination.py` | A small network learns one step of a damped pendulum from 20,000 random transitions, then imagines 1 to 100 steps ahead on its own predictions; the angle error against the real pendulum, 1,000 start states, 3 seeds. | CPU, about 10 seconds |
| `dreamer_budget.py` | How many steps one Dreamer-style update imagines at DreamerV3's default batch and horizon, and how long the same number of real steps would take. Plain arithmetic. | CPU, instant |
| `jepa_collapse.py` | Joint-embedding prediction on a toy world (8-number hidden state rendered into noisy, masked 64-number views): an untrained encoder, a shared encoder with no tricks, stop-gradient, an EMA target with and without a predictor, and VICReg regularizers, 3 seeds each; final loss, embedding spread and a linear probe's R^2. | CPU, about 2.5 minutes |

Run times measured on the M4 for this folder. The outputs here came from an Apple M4 laptop with torch 2.14.0, on the CPU.

```bash
pip install "numpy==2.*" "torch==2.14.*"
python pendulum_imagination.py
python dreamer_budget.py
python jepa_collapse.py
```
