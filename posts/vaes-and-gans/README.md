# VAEs and GANs: the full code

The scripts behind [VAEs and GANs: Two Ways to Sample What You Can't Write Down](https://residual-stream.lvtan71.workers.dev/posts/vaes-and-gans/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `elbo_check.py` | A 2-D linear-Gaussian latent model solved exactly: log p(x), the true posterior, and for four Gaussian q(z\|x) the sampled and exact reconstruction term, the KL to the prior, the ELBO, and its gap to log p(x) next to the KL to the true posterior. | CPU, instant |
| `gen2d.py` | The GAN's value at the optimal discriminator checked against 2 JS − log 4; then a VAE, a non-saturating GAN and WGAN-GP trained on 8 Gaussian modes on a ring, 6,000 generator steps, 3 seeds each, scored by modes covered and share of samples near a mode. | CPU, about 2 minutes |
| `vae_codes.py` | Retrains `gen2d.py`'s VAE for the same 3 seeds and scores decoded reconstructions, codes the encoder could give, and codes from the prior. Loads `gen2d.py`. | CPU, about 10 seconds |
| `wgan_long.py` | WGAN-GP seed 2 run to 12,000 generator steps and scored every 1,500. Loads `gen2d.py`. | CPU, about 45 seconds |

The whole folder runs in about 3 minutes on a CPU. The outputs here came from an Apple M4 laptop with torch 2.14.0, on its CPU. The exception is `gen2d.out`, regenerated on the same laptop's CPU with torch 2.14.1 to recover two lines missing from the saved output; its other lines match the original run.

```bash
pip install "numpy==2.*" "torch==2.14.*"
python elbo_check.py
python gen2d.py
python vae_codes.py
python wgan_long.py
```
