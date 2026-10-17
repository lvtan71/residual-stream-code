import torch
import torch.nn as nn

# Masked diffusion for text, on a toy language where every token depends
# on the others: the 90 sequences x_i = (a + i d) mod 10, i = 0..7, for a
# start a in 0..9 and a step d in 1..9 ("3 6 9 2 5 8 1 4"). Two adjacent
# digits pin down the rest. Token 10 is [MASK].
L, MASK = 8, 10
lang = torch.tensor([[(a + i * d) % 10 for i in range(L)] for a in range(10) for d in range(1, 10)])
valid = {tuple(r) for r in lang.tolist()}

class Denoiser(nn.Module):
    """A small bidirectional Transformer: every position sees every other,
    masked or not, and predicts a digit for each."""
    def __init__(self, d=64):
        super().__init__()
        self.tok, self.pos = nn.Embedding(11, d), nn.Parameter(0.02 * torch.randn(L, d))
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True)
        self.body, self.out = nn.TransformerEncoder(layer, 3), nn.Linear(d, 10)
    def forward(self, x):
        return self.out(self.body(self.tok(x) + self.pos))

def train(seed, steps=10000):
    """The masked-diffusion loss (MDLM, Sahoo et al. 2024): draw a masking
    rate t, mask each token with probability t, and score the model's
    cross-entropy on the masked tokens, weighted by 1/t."""
    torch.manual_seed(seed)
    net = Denoiser(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(steps):
        x0 = lang[torch.randint(0, len(lang), (256,))]
        t = torch.rand(256, 1).clamp(min=1 / L)
        m = torch.rand(256, L) < t
        logits = net(torch.where(m, MASK, x0))
        ce = nn.functional.cross_entropy(logits.transpose(1, 2), x0, reduction="none")
        loss = ((ce * m) / t).sum(1).mean() / L
        opt.zero_grad(); loss.backward(); opt.step()
    return net

@torch.no_grad()
def sample(net, n, k, order):
    """Start fully masked; each step unmasks k positions (k can be a list,
    one entry per step), sampling each from the model's distribution for
    that position, independently of the others unmasked in the same step."""
    x = torch.full((n, L), MASK)
    for k in (k if isinstance(k, list) else [k] * (L // k)):
        probs = net(x).softmax(-1)
        draw = torch.distributions.Categorical(probs).sample()
        free = x == MASK
        if order == "left to right":
            key = -torch.arange(L).float().expand(n, L)
        elif order == "random":
            key = torch.rand(n, L)
        else:                                               # "most confident": highest top probability first
            key = probs.max(-1).values
        pick = torch.where(free, key, -float("inf")).topk(k, -1).indices
        x.scatter_(1, pick, draw.gather(1, pick))
    return x

import itertools
def first_step_exact(k, order):
    """With everything masked, a perfect model's distribution at each
    position is uniform over the 10 digits, and one step samples the k
    chosen positions independently. The chance those k digits fit some
    valid sequence: valid patterns on those positions / 10^k, averaged
    over which positions the order picks."""
    sets = [tuple(range(k))] if order == "left to right" else list(itertools.combinations(range(L), k))
    return sum(len({tuple(r[i] for i in p) for r in lang.tolist()}) / 10 ** k for p in sets) / len(sets)

nets = [train(s) for s in range(3)]
print(f"{len(lang)} valid sequences of {L} digits; 3 models (seeds), 2,000 samples each\n")
print(f"{'tokens per step':>15s} {'steps':>6s} {'order':>15s}   valid samples, per seed     distinct valid (of 90)   first step, exact")
for k in (1, 2, 4, 8):
    for order in (("left to right", "random", "most confident") if k < L else ("all at once",)):
        torch.manual_seed(1)
        res = []
        for net in nets:
            s = [tuple(r) for r in sample(net, 2000, k, order).tolist()]
            ok = [r for r in s if r in valid]
            res.append((len(ok) / len(s), len(set(ok))))
        exact = f"{first_step_exact(k, 'random' if order == 'all at once' else order):8.2%}" if order != "most confident" else ""
        print(f"{k:15d} {L // k:6d} {order:>15s}   {'  '.join(f'{v:6.1%}' for v, _ in res)}      "
              f"{'  '.join(f'{d:3d}' for _, d in res)}   {exact:>16s}", flush=True)

# Parallel is safe once the context determines the rest: two digits one at
# a time, then the other six in a single step.
for order in ("left to right", "random"):
    torch.manual_seed(1)
    res = []
    for net in nets:
        s = [tuple(r) for r in sample(net, 2000, [1, 1, 6], order).tolist()]
        ok = [r for r in s if r in valid]
        res.append((len(ok) / len(s), len(set(ok))))
    print(f"{'1, 1, then 6':>15s} {3:6d} {order:>15s}   {'  '.join(f'{v:6.1%}' for v, _ in res)}      "
          f"{'  '.join(f'{d:3d}' for _, d in res)}", flush=True)

# Why random order fails here: two digits at distance g fix the step d only
# when g d = (difference) mod 10 has one solution, i.e. g is coprime to 10.
for g in range(1, L):
    n = sum(len({d for d in range(1, 10) if (g * d) % 10 == e}) == 1 for e in range(10)
            if any((g * d) % 10 == e for d in range(1, 10)))
    m = sum(any((g * d) % 10 == e for d in range(1, 10)) for e in range(10))
    print(f"digits {g} apart: {n} of the {m} possible differences fix the step", flush=True)