# Beyond the GPU: the full code

The scripts behind [Beyond the GPU: TPUs, LPUs and What Specialization Buys](https://residual-stream.lvtan71.workers.dev/posts/beyond-the-gpu-tpus-and-lpus/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `systolic.py` | A 3×3 output-stationary systolic array computing C = A @ B cycle by cycle: how many cells work each cycle, a check against A @ B, and how often each value read from memory is reused for n = 3, 128, 256. | CPU, instant |
| `sram.py` | How many chips it takes to hold a 70B model's weights at 16 and 8 bits on a GroqChip, an H100, a TPU v6e and a TPU7x, and the bandwidth-or-compute bound on one stream's tokens per second (vendor specs). | Python, instant |

The outputs here came from an Apple M4 laptop; the scripts need only numpy.

```bash
pip install "numpy==2.*"
python systolic.py
python sram.py
```
