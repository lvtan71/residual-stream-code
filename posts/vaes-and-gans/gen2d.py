import math
import numpy as np
import torch
import torch.nn as nn

# Three generative models on the same 2-D data: 8 Gaussian modes on a ring
# of radius 2 (std 0.05 each), the standard test for mode collapse. Each is
# scored on 2,000 samples: modes covered (a mode counts if more than 1% of
# samples fall within 3 std of it) and quality (the share of samples within
# 3 std of some mode). Three seeds each.
centers = torch.tensor([[2 * math.cos(a), 2 * math.sin(a)] for a in np.arange(8) * math.pi / 4], dtype=torch.float32)
def data(n, g):
    return centers[torch.randint(0, 8, (n,), generator=g)] + 0.05 * torch.randn(n, 2, generator=g)
def score(x):
    d = torch.cdist(x, centers)
    near = d.min(1).values < 0.15
    hits = torch.bincount(d.argmin(1)[near], minlength=8)
    return int((hits > 0.01 * len(x)).sum()), near.float().mean().item()
mlp = lambda i, o: nn.Sequential(nn.Linear(i, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, o))

def vae(seed, steps=6000, beta=1.0):
    """Encoder gives q(z|x) = N(mu, sigma^2) in 2-D; decoder gives p(x|z) =
    N(f(z), 0.05^2 I). Loss: minus the ELBO = reconstruction + KL to N(0, I)."""
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    enc, dec = mlp(2, 4), mlp(2, 2)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], 1e-3)
    for _ in range(steps):
        x = data(256, g)
        mu, logvar = enc(x).chunk(2, -1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()            # the reparameterization trick
        rec = ((dec(z) - x) ** 2).sum(-1) / (2 * 0.05 ** 2)
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(-1)
        loss = (rec + beta * kl).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return score(dec(torch.randn(2000, 2)))

def gan(seed, steps=6000, wgan=False):
    """Generator maps 2-D noise to points; the critic tells data from
    samples. Ordinary GAN: the non-saturating loss of Goodfellow et al.
    WGAN-GP: a critic score difference with a gradient penalty."""
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    G, D = mlp(2, 2), mlp(2, 1)
    og = torch.optim.Adam(G.parameters(), 1e-4 if wgan else 2e-4, betas=(0.5, 0.9))
    od = torch.optim.Adam(D.parameters(), 1e-4 if wgan else 2e-4, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss()
    for step in range(steps):
        for _ in range(5 if wgan else 1):
            x, fake = data(256, g), G(torch.randn(256, 2)).detach()
            if wgan:
                eps = torch.rand(256, 1)
                mix = (eps * x + (1 - eps) * fake).requires_grad_(True)
                grad = torch.autograd.grad(D(mix).sum(), mix, create_graph=True)[0]
                ld = D(fake).mean() - D(x).mean() + 10 * ((grad.norm(dim=1) - 1) ** 2).mean()
            else:
                ld = bce(D(x), torch.ones(256, 1)) + bce(D(fake), torch.zeros(256, 1))
            od.zero_grad(); ld.backward(); od.step()
        fake = G(torch.randn(256, 2))
        lg = -D(fake).mean() if wgan else bce(D(fake), torch.ones(256, 1))
        og.zero_grad(); lg.backward(); og.step()
    with torch.no_grad():
        return score(G(torch.randn(2000, 2)))

# The GAN's value at the optimal discriminator is 2 JS(p, q) - log 4: checked
# for two 1-D Gaussians on a grid.
xs = np.linspace(-10, 10, 20001); dx = xs[1] - xs[0]
p = np.exp(-0.5 * xs ** 2) / math.sqrt(2 * math.pi); q = np.exp(-0.5 * (xs - 1.5) ** 2) / math.sqrt(2 * math.pi)
Dstar = p / (p + q)
value = (p * np.log(Dstar) + q * np.log(1 - Dstar)).sum() * dx
m = (p + q) / 2
js = 0.5 * ((p * np.log(p / m)).sum() + (q * np.log(q / m)).sum()) * dx
print(f"optimal-discriminator value {value:.4f}; 2 JS - log 4 = {2 * js - math.log(4):.4f}")

print(f"\n{'model':24s} modes covered (of 8), per seed   share of samples near a mode")
for name, fn in (("VAE", vae), ("GAN (non-saturating)", gan), ("WGAN-GP", lambda s: gan(s, wgan=True))):
    res = [fn(s) for s in range(3)]
    print(f"{name:24s} {'  '.join(str(r[0]) for r in res):>18s}            {'  '.join(f'{r[1]:.0%}' for r in res)}", flush=True)
