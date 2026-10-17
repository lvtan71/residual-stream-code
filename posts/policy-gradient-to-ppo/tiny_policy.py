import numpy as np
import torch
import torch.nn.functional as F

# Policy gradient on the smallest "language model" that still has the
# structure: a policy writes 3 tokens from a vocabulary of 8, one at a time,
# each conditioned on the tokens before it (a table of logits per prefix,
# 1 + 8 + 64 = 73 rows). The reward, given only at the end, counts positions
# matching a hidden target, plus 10 (rewards are rarely centered at zero).
# Learners, all with the same samples per update (a batch of 16 sequences):
#   REINFORCE:            grad = mean of R * grad log pi(sequence);
#   the same, with the +10 removed (rewards 0 to 3), the counterfactual;
#   REINFORCE + baseline: (R - mean R of the batch) instead of R;
#   the same, 4 passes over each batch, the control for what follows;
#   PPO:                  the same advantage, 4 passes over each batch with
#                         the clipped ratio objective (clip 0.2), simplified:
#                         one ratio per sequence, and the batch-mean baseline
#                         instead of a value head.
V, L, BATCH, TARGET = 8, 3, 16, (5, 2, 7)

def rows(seq):
    """Index of the logit row for each step: the prefix before that token."""
    out, r = [], 0
    for t, tok in enumerate(seq):
        out.append(r)
        r = 1 + tok if t == 0 else 1 + V + seq[0] * V + tok
    return out

def sample(logits, g):
    seqs = []
    for _ in range(BATCH):
        seq = []
        for t in range(L):
            r = rows(seq + [0])[t]
            seq.append(int(torch.multinomial(logits[r].softmax(-1), 1, generator=g)))
        seqs.append(seq)
    return seqs

def logprob(logits, seqs):
    idx = torch.tensor([rows(s) for s in seqs])                 # (BATCH, L)
    tok = torch.tensor(seqs)
    return logits[idx].log_softmax(-1).gather(-1, tok[..., None])[..., 0].sum(-1)

def run(method, seed, updates=300, lr=0.1):
    g = torch.Generator().manual_seed(seed)
    logits = torch.zeros(1 + V + V * V, V, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    curve = []
    global clipped
    for _ in range(updates):
        seqs = sample(logits.detach(), g)
        R = torch.tensor([10.0 + sum(a == b for a, b in zip(s, TARGET)) for s in seqs])
        curve.append(R.mean().item() - 10)
        adv = {"REINFORCE": R, "REINFORCE, no offset": R - 10}.get(method, R - R.mean())
        old = logprob(logits, seqs).detach()
        for epoch in range(1 if method in ("REINFORCE", "REINFORCE, no offset", "REINFORCE + baseline") else 4):
            lp = logprob(logits, seqs)
            if method == "PPO":
                ratio = (lp - old).exp()
                if epoch > 0:
                    clipped.append(((ratio < 0.8) | (ratio > 1.2)).float().mean().item())
                loss = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            else:
                loss = -(adv * lp).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return np.array(curve)

clipped = []
SEEDS = range(10)
for lr in (0.1, 0.03):
    print(f"Adam learning rate {lr}: mean matches per sequence (of 3) over {len(SEEDS)} seeds, at update")
    print(f"{'':22s} " + "".join(f"{u:>7d}" for u in (10, 50, 100, 200, 300)) + "   seeds solved (>= 2.9 at the end)")
    for method in ("REINFORCE", "REINFORCE, no offset", "REINFORCE + baseline", "baseline, 4 passes", "PPO"):
        curves = np.stack([run(method, seed, lr=lr) for seed in SEEDS])
        at = lambda u: curves[:, max(0, u - 10):u].mean()      # average of the 10 updates up to u
        solved = (curves[:, -10:].mean(1) >= 2.9).sum()
        print(f"{method:22s} " + "".join(f"{at(u):7.2f}" for u in (10, 50, 100, 200, 300)) + f"   {solved} of {len(SEEDS)}")
    print(f"PPO's clip was active for {np.mean(clipped):.0%} of sequences in passes 2 to 4")
    clipped.clear()

# Why the offset hurts even with Adam, which rescales steps: the direction
# of one batch's gradient, against the exact gradient (all 512 sequences
# enumerated), at the start of training. Mean cosine over 200 batches.
logits = torch.zeros(1 + V + V * V, V, requires_grad=True)
every = [list(s) for s in np.ndindex(V, V, V)]
lp_all = logprob(logits, every)
match = torch.tensor([float(sum(a == b for a, b in zip(s, TARGET))) for s in every])
exact = torch.autograd.grad((lp_all.exp() * match).sum(), logits)[0].flatten()
g = torch.Generator().manual_seed(0)
for name, shift in (("rewards 10 to 13", 10.0), ("rewards 0 to 3", 0.0), ("minus the batch mean", None)):
    cos = []
    for _ in range(200):
        seqs = sample(logits.detach(), g)
        R = torch.tensor([float(sum(a == b for a, b in zip(s, TARGET))) for s in seqs])
        w = R + shift if shift is not None else R - R.mean()
        grad = torch.autograd.grad((w * logprob(logits, seqs)).mean(), logits)[0].flatten()
        cos.append(F.cosine_similarity(grad, exact, 0).item())
    print(f"one batch's gradient vs the exact one, {name:20s}: cosine {np.mean(cos):.2f}")