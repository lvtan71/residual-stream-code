# Attention Alternatives: the full code

The scripts behind [Attention Alternatives: Sparse, Linear, and Recurrent](https://residual-stream.lvtan71.workers.dev/posts/attention-alternatives/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `bills.py` | Prefill FLOPs, the KV cache of a Llama-3-8B shape, the linear-attention state, the n/d saving and a 4,096-token sliding window, as arithmetic. | Python, instant |
| `linear_attention.py` | Linear attention computed as an n x n product, as a reassociated d x d product and as a recurrence, checked equal; and the rank of softmax's similarity matrix. | numpy, instant |
| `order_timing.py` | Times (QK^T)V against Q(K^T V) from 32 to 16,384 tokens. | torch; GPU if there is one. Time it with nothing else running |
| `recall.py` | Writes n random key-value pairs into a KV cache, a linear-attention state and a delta-rule state, and reads each back: by n, by state width, and by when each pair was written. | numpy, under a second |
| `ssm.py` | A diagonal state space model: discretization, recurrence against convolution, input-dependent parameters, a parallel scan, and Mamba-2's quadratic form. | numpy, instant |
| `configs.py` | Reads the attention layout of thirteen released models from their config.json at pinned revisions, and the KV cache their attention layers need at 256K tokens. | `huggingface_hub`; downloads only config files |

The outputs came from an Apple M4 laptop (10 cores) with Python 3.12, numpy 2.5.4 and torch 2.14.1. `order_timing.out` was measured on the M4's GPU (MPS) with nothing else running.

```bash
pip install numpy "torch==2.14.*" huggingface_hub
python bills.py
python linear_attention.py
python order_timing.py
python recall.py
python ssm.py
python configs.py
```
