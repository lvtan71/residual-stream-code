# DPO: the full code

The scripts behind [DPO: The Policy Is Its Own Reward Model](https://residual-stream.lvtan71.workers.dev/posts/dpo-direct-preference-optimization/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `dpo_toy.py` | DPO on the exact toy world of the RLHF post: 512 responses, a hidden gold reward, 300 Bradley–Terry comparisons. First the policy family of RLHF's reward model, where DPO and RLHF solved to convergence give the same policy; then one free logit per response, trained with DPO (Adam and SGD), DPO with soft labels, and IPO at two values of β. | CPU, about 15 seconds |

`rlhf_toy.py` is the script of [RLHF: From Christiano 2017 to InstructGPT to Today's Recipe](https://residual-stream.lvtan71.workers.dev/posts/rlhf-from-comparisons/), copied here; `dpo_toy.py` runs its definitions (everything before its experiment loop) to get the same `world`, `X`, `Y` and `sigmoid`.

The outputs here came from an Apple M4 laptop with torch 2.14.0. `dpo_toy.py` runs on the CPU in float64 and reproduces its `.out` exactly.

```bash
pip install "torch==2.14.*" "numpy==2.*"
python dpo_toy.py
```
