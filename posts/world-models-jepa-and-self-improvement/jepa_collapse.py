import copy
import torch
import torch.nn as nn

# Joint-embedding prediction on a toy world. Each sample has a hidden state
# s (8 numbers) that a fixed random network renders into 64-number
# "observations"; two views of the same sample differ by heavy noise and a
# random mask (the context sees 50% of the observation, the target another
# draw). An encoder embeds each view; a predictor maps the context's
# embedding to the target's; the loss is their squared distance. Nothing
# in the loss forbids the trivial answer: embed everything as the same
# point. Five variants, three seeds each, against an untrained encoder.
D_S, D_X, D_Z = 8, 64, 32
torch.manual_seed(1234)
render = nn.Sequential(nn.Linear(D_S, 128), nn.Tanh(), nn.Linear(128, D_X)).requires_grad_(False)
def views(n):
    s = torch.randn(n, D_S)
    x = render(s)
    view = lambda: (x + 0.3 * torch.randn_like(x)) * (torch.rand_like(x) < 0.5)
    return s, view(), view()
mlp = lambda i, o: nn.Sequential(nn.Linear(i, 256), nn.ReLU(), nn.Linear(256, o))

def probe(z, s):
    """How much of the hidden state a linear readout recovers from the
    embedding (R^2 on held-out samples): a collapsed embedding scores 0."""
    z = torch.cat([z, torch.ones(len(z), 1)], 1)
    w = torch.linalg.lstsq(z[:2000], s[:2000]).solution
    err = ((z[2000:] @ w - s[2000:]) ** 2).mean()
    return 1 - err / s[2000:].var(0).mean()

def vicreg_terms(z):
    """VICReg's regularizers: keep each embedding dimension's spread across
    the batch above 1, and the dimensions uncorrelated."""
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    cov = (z.T @ z) / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return 25 * torch.relu(1 - std).mean() + (off ** 2).sum() / z.shape[1]

def run(variant, seed, steps=10000):
    torch.manual_seed(seed)
    enc, pred = mlp(D_X, D_Z), mlp(D_Z, D_Z)
    target_enc = copy.deepcopy(enc).requires_grad_(False)
    params = list(enc.parameters()) + (list(pred.parameters()) if variant != "EMA, no predictor" else [])
    opt = torch.optim.Adam(params, 1e-3)
    for _ in range(steps if variant != "untrained encoder" else 0):
        _, ctx, tgt = views(256)
        z_ctx = enc(ctx)
        if variant in ("shared encoder, no tricks", "VICReg regularizers"):
            z_tgt = enc(tgt)                                  # gradients flow into both sides
        elif variant == "stop-gradient":
            z_tgt = enc(tgt).detach()                         # the target side is treated as a constant
        else:                                                 # EMA: a slow copy of the encoder makes targets
            z_tgt = target_enc(tgt)
        p = z_ctx if variant == "EMA, no predictor" else pred(z_ctx)
        loss = ((p - z_tgt) ** 2).sum(1).mean()
        total = loss + (vicreg_terms(z_ctx) + vicreg_terms(z_tgt) if variant == "VICReg regularizers" else 0)
        opt.zero_grad(); total.backward(); opt.step()
        if variant.startswith("EMA"):
            with torch.no_grad():
                for a, b in zip(target_enc.parameters(), enc.parameters()):
                    a.mul_(0.996).add_(0.004 * b)
    with torch.no_grad():
        s, ctx, tgt = views(4000)
        z = enc(ctx)
        last = ((pred(z) - enc(tgt)) ** 2).sum(1).mean().item() if variant == "untrained encoder" else loss.item()
        return last, z.std(0).mean().item(), probe(z, s).item()

print(f"{'variant':28s} {'final loss':>10s} {'embedding std':>14s} {'probe R^2':>10s}   (mean of 3 seeds, 10,000 steps)")
for variant in ("untrained encoder", "shared encoder, no tricks", "stop-gradient", "EMA target + predictor",
                "EMA, no predictor", "VICReg regularizers"):
    r = torch.tensor([run(variant, seed) for seed in range(3)]).mean(0)
    print(f"{variant:28s} {r[0]:10.4f} {r[1]:14.6f} {r[2]:10.2f}", flush=True)