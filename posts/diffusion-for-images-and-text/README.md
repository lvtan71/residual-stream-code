# Diffusion, for Images and for Text: the full code

The scripts behind [Diffusion, for Images and for Text: Learn to Undo a Little Noise](https://residual-stream.lvtan71.workers.dev/posts/diffusion-for-images-and-text/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `diff2d.py` | The linear noise schedule, then a noise-predicting network on the eight-mode ring of [VAEs and GANs](https://residual-stream.lvtan71.workers.dev/posts/vaes-and-gans/), trained on three seeds with a label dropped 10% of the time: DDPM and DDIM at 100 down to 2 steps, and classifier-free guidance at w = 0, 1, 3, 7. Saves the networks to `nets.pt`. | CPU, about 3 minutes (estimate) |
| `ddim_spacing.py` | DDIM at 5, 10 and 20 steps with three choices of which timesteps to visit. Reads `diff2d.py`'s definitions and `nets.pt`. | CPU, under a minute |
| `guidance_radius.py` | Where guided samples go: median distance from the ring's center and spread around the mode asked for, at w = 1, 3, 7. Reads `diff2d.py`'s definitions and `nets.pt`. | CPU, a few seconds |
| `masked_diffusion.py` | Masked diffusion on 90 eight-digit arithmetic sequences: a small bidirectional Transformer trained on three seeds, then decoded 1, 2, 4 or 8 tokens per step in three orders, against the exact first-step chance. | CPU, about 1.5 minutes (estimate) |
| `flops.py` | Arithmetic: FLOPs and the latency floor of autoregressive decoding against masked diffusion for an 8B model on one H100. | CPU, instant |

The whole folder runs in about 5 minutes on a CPU; the per-script times are estimates from that total. The outputs here came from an Apple M4 laptop with torch 2.14.0. Other torch versions or CPUs can change the last digit of some percentages.

```bash
pip install "numpy==2.*" "torch==2.14.*"
python diff2d.py
python ddim_spacing.py
python guidance_radius.py
python masked_diffusion.py
python flops.py
```
