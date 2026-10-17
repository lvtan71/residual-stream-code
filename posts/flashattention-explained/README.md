# FlashAttention: the full code

The scripts behind [FlashAttention: Exact Attention at the Speed of Memory](https://residual-stream.lvtan71.workers.dev/posts/flashattention-explained/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `traffic.py` | Counts the HBM traffic of one attention head's forward pass, standard against FlashAttention, by sequence length and row-block size, and what each keeps for the backward pass. | CPU, instant |
| `online_softmax.py` | The online softmax in numpy: the post's worked example block by block, then tiled attention against standard attention on random inputs. | numpy, instant |
| `sdpa_vs_standard.py` | Times one head of three-step attention against PyTorch's fused `scaled_dot_product_attention` on the CPU at 2,048, 8,192 and 16,384 tokens, with each run's peak memory, in a fresh process per measurement. | CPU, under a minute; about 2.3 GB of RAM at 16,384 tokens |

The whole folder runs in about a minute on a CPU. The outputs here came from an Apple M4 laptop with torch 2.11.0 and Python 3.12.13 for `sdpa_vs_standard.py`; `online_softmax.py` and `traffic.py` ran under the system Python 3.9.6 with numpy 2.0.2. Timings and peak memory depend on the machine; the comparison between the two methods is what carries over. Run `sdpa_vs_standard.py` on an otherwise idle machine.

```bash
pip install "numpy==2.*" "torch==2.11.*"
python traffic.py
python online_softmax.py
python sdpa_vs_standard.py
```
