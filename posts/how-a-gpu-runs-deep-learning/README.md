# How a GPU Runs Deep Learning: the full code

The scripts behind [How a GPU Runs Deep Learning: Memory, the Roofline, and a First Kernel](https://residual-stream.lvtan71.workers.dev/posts/how-a-gpu-runs-deep-learning/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `kernels.metal` | Every kernel in the post, in Metal: vector add, strided copy, and the matrix multiply as naive, tiled 16 x 16, tiled 32 x 32 and with register tiles. | compiled by `gpu_util.py` |
| `gpu_util.py` | Compiles `kernels.metal` and defines the two timers: one call at a time, and many calls queued with one wait. | helper, no output |
| `bandwidth.py` | Memory bandwidth from the vector add and PyTorch's copy, and how a strided copy loses it (coalescing). | Apple-silicon GPU, under a minute |
| `roofline.py` | Vector add, matrix-vector and matrix multiplies of growing size, timed: intensity, GFLOP/s and GB/s on the roofline. | Apple-silicon GPU, under a minute |
| `matmul_ladder.py` | The four hand-written matmul kernels against PyTorch's, at n = 512, 1024, 2048, each checked against a float64 reference. | Apple-silicon GPU, under a minute |
| `tile_intensity.py` | Arithmetic: reads per output and intensity for each tile size, and the two ridge points. | CPU, instant |
| `smem_probe.py` | Compiles a probe kernel with 16, 32 and 64 KB of threadgroup memory to find this GPU's limit. | Apple-silicon GPU, instant |
| `waves.py` | Arithmetic: 128 x 128 tiles in waves over an H100's 132 SMs, and how much of each wave is useful. | CPU, instant |

The kernels are written in Metal and run through `torch.mps.compile_shader`, so every script that imports `gpu_util.py`, and `smem_probe.py`, needs an Apple-silicon Mac; they do not run on CUDA or a CPU. `tile_intensity.py` and `waves.py` are only arithmetic and run anywhere. The whole folder runs in about 2 minutes; the per-script times are estimates from that total. The outputs here came from an Apple M4 laptop with torch 2.11.0 (Python 3.12.13). Timings depend on the machine; time with nothing else using the GPU.

```bash
pip install "torch==2.11.*"
python bandwidth.py
python roofline.py
python matmul_ladder.py
python tile_intensity.py
python smem_probe.py
python waves.py
```
