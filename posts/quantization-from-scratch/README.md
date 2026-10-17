# Quantization from Scratch: the full code

The scripts behind [Quantization from Scratch: Scale, Zero-Point, GPTQ and AWQ](https://residual-stream.lvtan71.workers.dev/posts/quantization-from-scratch/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `capture.py` | Runs Qwen2.5-0.5B (base) over the blog's text and records the inputs and weight of the MLP down-projection in layers 12 and 22: `X_<L>.npy` (4,096 tokens), `W_<L>.npy`, and from 32,768 later tokens only GPTQ's `H_big_<L>.npy` and AWQ's `mag_big_<L>.npy`. Not shown in the post, and it has no `.out`: the post's inputs were captured before the post's code was saved. `capture.py` is a reconstruction: it follows the logic of the original capture scripts, which covered layer 12 only and read the model and text from local paths. | CPU, about a minute (estimate); downloads the model (about 1 GB) at a pinned revision. Reads `blog_train.txt`, a link to `../../data/blog_train.txt` |
| `measure.py` | Round-to-nearest (per tensor, per channel, per group of 128; asymmetric, symmetric, NF4), GPTQ and AWQ (scaling, then clipping), implemented in numpy and measured as the layer's output error on 2,048 held-out tokens; and GPTQ with too little calibration. | CPU, about 2 minutes (estimate); reads the `.npy` files `capture.py` writes |
| `decode_speed.py` | A roofline bound on one decode step of an 8B model on an H100, BF16 against INT4 weights, by batch size. | Python, instant |

The whole folder runs in about 3 minutes on a CPU; the per-script times are estimates from that total. The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.17.*" huggingface_hub
python capture.py
python measure.py
python decode_speed.py
```
