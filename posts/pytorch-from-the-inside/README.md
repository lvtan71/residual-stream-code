# PyTorch from the Inside: the full code

The scripts behind [PyTorch from the Inside: Tensors, Autograd and Precision](https://residual-stream.lvtan71.workers.dev/posts/pytorch-from-the-inside/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `tensor_views.py` | A tensor's shape, strides and offset; a transpose and a slice as views of the same memory; `view` failing where `reshape` copies. | CPU, instant |
| `tiny_autograd.py` | A 55-line autograd tape on NumPy: add, matmul, ReLU, log-loss, broadcasting. A helper; prints nothing. | |
| `check_autograd.py` | `tiny_autograd.py` against PyTorch on the XOR worked example from "Backpropagation, by Hand", then on 200 random networks with a shared weight and a broadcast bias. | CPU, a few seconds |
| `graph.py` | Walks the graph PyTorch records for a two-layer network, counts its nodes when a layer is applied once and three times, and shows that `no_grad` records nothing. | CPU, instant |
| `timing.py` | Matrix-multiply TFLOP/s on the CPU and the GPU in FP32, FP16 and BF16; a GPU multiply timed with and without synchronizing; per-call cost against 200 queued calls. | A GPU (CUDA or Apple `mps`), about 20 seconds. Timings depend on the machine; run it with nothing else busy |
| `float_formats.py` | FP32, FP16 and BF16 side by side: range, smallest values, step at 1.0, rounding examples, 10,000 small updates, and loss scaling for FP16 gradients. | CPU, a few seconds |
| `router_torch.py` | The four-way router from "Cross-Entropy and the Softmax Gradient" trained with `torch.optim.AdamW`, next to the hand-written AdamW from "Initialization and Optimizers" on the same mini-batches. | CPU, a few seconds |
| `step_memory.py` | Bytes per parameter kept by an AdamW training step under `torch.autocast` (BF16 on the CPU): weights, gradients, Adam state, and BF16 weight copies saved for backward. | CPU, a few seconds |
| `fusion.py` | Two chains of elementwise operations, eager against `torch.compile`, next to the time of one read and one write (`x.clone()`). | CPU, about 15 seconds; `torch.compile` needs a C++ compiler |

Helpers copied from other posts: `router_data.py` (from "Features and Overfitting"), `router4.py` and `softmax_basics.py` (from "Cross-Entropy and the Softmax Gradient"), `optim.py` (from "Initialization and Optimizers"). `router-questions.csv` links to `../../data/router-questions.csv`, the blog's labeled reader questions.

The whole folder runs in about a minute; per-script run times are estimates. The outputs here came from an Apple M4 laptop with torch 2.11.0 (Python 3.12.13): `timing.py` ran on its GPU through `mps`, the rest on its CPU. The words "on mps" in `timing.py`'s last two lines are fixed text.

```bash
pip install "numpy==2.*" "torch==2.11.*"
python tensor_views.py
python check_autograd.py
python graph.py
python timing.py
python float_formats.py
python router_torch.py
python step_memory.py
python fusion.py
```
