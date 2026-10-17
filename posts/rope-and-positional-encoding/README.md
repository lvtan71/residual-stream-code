# Where Am I? Position in Attention, from Sinusoids to RoPE: the full code

The scripts behind [Where Am I? Position in Attention, from Sinusoids to RoPE](https://residual-stream.lvtan71.workers.dev/posts/rope-and-positional-encoding/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `permutation_rope.py` | Checks that attention without positions is permutation-equivariant and that a causal mask breaks this; RoPE as a rotation of dimension pairs, whose query-key score depends only on the distance and which keeps a vector's length. | CPU, instant (numpy) |
| `train_position.py` | One run: a 2-layer causal Transformer with one position scheme (`none`, `sinusoid`, `learned`, `rope`, `rope:<base>`, `alibi:<slope>`), trained 2,000 steps on length 32 for the `lag` or `seen` task, then tested at lengths 32, 64 and 128, past position 32, and with log-length score scaling. Arguments: scheme, seed, task. | CPU, about 10 seconds a run |
| `all_runs.py` | Runs `train_position.py` for both tasks, seven schemes and seeds 0 and 1: 28 runs, one line each. | CPU, about 6 minutes |
| `wavelengths.py` | Wavelength of each RoPE dimension pair at head dimension 128, for bases 10,000 and 500,000, and how many pairs never complete a turn within 8,192 and 131,072 positions. | CPU, instant (numpy) |

The per-run time is an estimate from the measured total of about 6 minutes for the 28 runs. The outputs here came from an Apple M4 laptop with torch 2.11.0 (Python 3.12.13), on the CPU (`train_position.py` uses two threads).

```bash
pip install "numpy==2.*" "torch==2.11.*"
python permutation_rope.py
python all_runs.py
python wavelengths.py
```
