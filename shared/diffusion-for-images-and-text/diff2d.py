import math
import numpy as np
import torch
import torch.nn as nn

# A diffusion model on "VAEs and GANs"'s test: 8 Gaussian modes on a ring of radius
# 2, std 0.05, scored the same way (modes covered, share of samples within
# 3 std of a mode). One network predicts the noise; it is trained with the
# mode as a label that is dropped 10% of the time, so the same network
# samples unconditionally (label dropped) and with guidance.
centers = torch.tensor([[2 * math.cos(a), 2 * math.sin(a)] for a in np.arange(8) * math.pi / 4], dtype=torch.float32)
def data(n, g):
    c = torch.randint(0, 8, (n,), generator=g)
    return centers[c] + 0.05 * torch.randn(n, 2, generator=g), c
def score(x):
    d = torch.cdist(x, centers)
    near = d.min(1).values < 0.15
    hits = torch.bincount(d.argmin(1)[near], minlength=8)
    return int((hits > 0.01 * len(x)).sum()), near.float().mean().item()

# The forward process: q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I),
# with the linear schedule of Ho et al. (2020).
T = 1000
beta = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
abar = torch.cumprod(1 - beta, 0)
print("linear schedule, beta from 1e-4 to 0.02 over T = 1000 steps")
print(f"{'t':>5s} {'abar_t':>9s} {'signal sqrt(abar)':>18s} {'noise sqrt(1-abar)':>19s} {'SNR':>9s}")
for t in (1, 10, 100, 250, 500, 750, 1000):
    a = abar[t - 1].item()
    print(f"{t:5d} {a:9.5f} {math.sqrt(a):18.4f} {math.sqrt(1 - a):19.4f} {a / (1 - a):9.4g}")
abar = abar.float()

class Eps(nn.Module):
    """eps_theta(x_t, t, label): an MLP on the point, a sinusoidal embedding
    of t, and a learned embedding of the label (index 8 = no label)."""
    def __init__(self, h=256):
        super().__init__()
        self.label = nn.Embedding(9, 64)
        self.net = nn.Sequential(nn.Linear(2 + 64 + 64, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
    def forward(self, x, t, c):
        f = torch.exp(-math.log(1000) * torch.arange(32) / 32)
        a = t[:, None].float() * f
        return self.net(torch.cat([x, a.sin(), a.cos(), self.label(c)], -1))

def train(seed, steps=20000):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    net = Eps(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x0, c = data(512, g)
        c = torch.where(torch.rand(512, generator=g) < 0.1, 8, c)             # drop the label 10% of the time
        t = torch.randint(0, T, (512,), generator=g)
        eps = torch.randn(512, 2, generator=g)
        a = abar[t][:, None]
        loss = ((net(a.sqrt() * x0 + (1 - a).sqrt() * eps, t, c) - eps) ** 2).mean()   # the DDPM loss
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net

@torch.no_grad()
def guided(net, x, t, c, w):
    """Classifier-free guidance: eps_uncond + w (eps_cond - eps_uncond).
    w = 1 is the plain conditional model; c = 8 means unconditional."""
    tt = torch.full((len(x),), t)
    if w == 1 or (c == 8).all():
        return net(x, tt, c)
    e_u = net(x, tt, torch.full_like(c, 8))
    return e_u + w * (net(x, tt, c) - e_u)

@torch.no_grad()
def ddpm(net, n, c=8, w=1.0):
    """Ancestral sampling, all T steps: x_{t-1} = mean + sqrt(beta_t) z."""
    x, c = torch.randn(n, 2), torch.full((n,), c) if isinstance(c, int) else c
    for t in reversed(range(T)):
        e = guided(net, x, t, c, w)
        x = (x - beta[t].float() / (1 - abar[t]).sqrt() * e) / (1 - beta[t].float()).sqrt()
        if t > 0:
            x = x + beta[t].float().sqrt() * torch.randn_like(x)
    return x

@torch.no_grad()
def ddim(net, n, steps, c=8, w=1.0):
    """DDIM with eta = 0 (Song et al., 2021): deterministic, and free to skip
    steps. Predict x_0 from the noise estimate, then jump to the next t."""
    x, c = torch.randn(n, 2), torch.full((n,), c) if isinstance(c, int) else c
    ts = torch.linspace(T - 1, 0, steps).round().long().tolist()
    for i, t in enumerate(ts):
        e = guided(net, x, t, c, w)
        x0 = (x - (1 - abar[t]).sqrt() * e) / abar[t].sqrt()
        a_next = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * e
    return x

nets = [train(s) for s in range(3)]
print(f"\nunconditional samples (2,000 per run); modes covered, then share near a mode, per seed")
for name, fn in [("DDPM, 1000 steps", lambda m: ddpm(m, 2000))] + \
                [(f"DDIM, {k} steps", lambda m, k=k: ddim(m, 2000, k)) for k in (100, 50, 20, 10, 5, 2)]:
    torch.manual_seed(100)
    res = [score(fn(m)) for m in nets]
    print(f"{name:18s} {'  '.join(str(r[0]) for r in res):>9s}    {'  '.join(f'{r[1]:4.0%}' for r in res)}", flush=True)

print(f"\nclassifier-free guidance, DDIM 50 steps, 250 samples asked for each mode (2,000 per run)")
print(f"{'w':>4s} {'landed on the mode asked for':>30s} {'share near any mode':>21s}")
asked = torch.arange(8).repeat_interleave(250)
for w in (0.0, 1.0, 3.0, 7.0):
    torch.manual_seed(200)
    on, near = [], []
    for m in nets:
        x = ddim(m, 2000, 50, c=asked, w=w)
        d = torch.cdist(x, centers)
        on.append(((d.argmin(1) == asked) & (d.min(1).values < 0.15)).float().mean().item())
        near.append(score(x)[1])
    print(f"{w:4.0f} {'  '.join(f'{v:4.0%}' for v in on):>30s} {'  '.join(f'{v:4.0%}' for v in near):>21s}", flush=True)
torch.save([m.state_dict() for m in nets], "nets.pt")
