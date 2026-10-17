import torch
# Where do the VAE's off-mode samples come from? gen2d.py's vae(), trained
# the same way, but returning the encoder and decoder instead of a score.
src = open("gen2d.py").read().split("# The GAN's value")[0]
ns = {}; exec(src.replace("return score(dec(torch.randn(2000, 2)))", "return enc, dec"), ns)
print(f"{'seed':>4s} {'reconstructions':>16s} {'z from the encoders':>20s} {'z from the prior':>17s}")
for seed in range(3):
    enc, dec = ns["vae"](seed)
    with torch.no_grad():
        x = ns["data"](2000, torch.Generator().manual_seed(99))
        mu, logvar = enc(x).chunk(2, -1)
        rec = dec(mu)                                                    # decode each point's mean code
        agg = dec(mu + torch.randn_like(mu) * (0.5 * logvar).exp())      # a code the encoder could give
        prior = dec(torch.randn(2000, 2))                                # what sampling the VAE does
    print(f"{seed:4d} " + "".join(f"{ns['score'](v)[1]:>{w}.0%}" for v, w in ((rec, 16), (agg, 21), (prior, 18))), flush=True)