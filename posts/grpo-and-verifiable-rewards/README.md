# GRPO and Reinforcement Learning from Verifiable Rewards: the full code

The scripts behind [GRPO and Reinforcement Learning from Verifiable Rewards](https://residual-stream.lvtan71.workers.dev/posts/grpo-and-verifiable-rewards/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `length_bias.py` | A toy with a checkable answer and free length: a table policy trained by GRPO, Dr. GRPO, and the two hybrids that change one normalization each, with groups of 4 and 16, 5 seeds, 400 updates. Prints accuracy and the mean length of right and wrong answers at updates 1, 50, 100 and 400. | CPU, about 20 seconds |

The outputs here came from an Apple M4 laptop with torch 2.14.0, on its CPU.

```bash
pip install "numpy==2.*" "torch==2.14.*"
python length_bias.py
```
