# Reinforcement Learning for Language Models: the full code

The scripts behind [Reinforcement Learning for Language Models: From Policy Gradient to PPO](https://residual-stream.lvtan71.workers.dev/posts/policy-gradient-to-ppo/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `tiny_policy.py` | A table policy that writes 3 tokens from a vocabulary of 8, rewarded for matching a hidden target: REINFORCE with and without the reward offset, REINFORCE with a batch-mean baseline, the baseline with 4 passes per batch, and a simplified PPO, 10 seeds each at two Adam learning rates. Then one batch's gradient against the exact gradient over all 512 sequences. | CPU, about 30 seconds |

The outputs here came from an Apple M4 laptop with torch 2.14.0. Seeds are fixed.

```bash
pip install numpy "torch==2.14.*"
python tiny_policy.py
```
