import numpy as np
import torch

# GRPO on a toy with checkable answers and free length. 16 "questions";
# the answer is a sequence of up to 8 tokens from {0, 1, 2, 3}, ended by an
# end token (or cut off at 8). It is correct if its first token equals the
# question's hidden answer: reward 1, else 0. Length never affects the
# reward, so any change in length comes from the loss alone. The policy is
# a table of logits per (question, position, previous token).
# Per update: 16 questions x a group of G answers each.
#   GRPO:     advantage (R - group mean) / group std; each answer's loss is
#             its tokens' -A log pi averaged over its own length, then
#             averaged over the group (Shao et al., 2024);
#   Dr. GRPO: advantage R - group mean; token losses summed and divided by
#             a constant, the maximum length (Liu et al., 2025);
# and two hybrids that change one of the two at a time. Length counts the
# end token, so an answer that ends at once has length 1 (and is wrong).
Q, V, LMAX, END = 16, 4, 8, 4

def run(method, G, seed, updates=400, lr=0.05):
    g = torch.Generator().manual_seed(seed)
    answer = torch.randint(0, V, (Q,), generator=g)
    logits = torch.zeros(Q, LMAX, V + 2, V + 1, requires_grad=True)   # prev token index V+1 = "start"
    opt = torch.optim.Adam([logits], lr=lr)
    hist = []
    for _ in range(updates):
        with torch.no_grad():                                         # sample Q x G answers
            q = torch.arange(Q).repeat_interleave(G)
            toks = torch.full((Q * G, LMAX), END)
            alive = torch.ones(Q * G, dtype=torch.bool)
            prev = torch.full((Q * G,), V + 1)
            length = torch.zeros(Q * G, dtype=torch.long)
            for t in range(LMAX):
                p = logits[q, t, prev].softmax(-1)
                a = torch.multinomial(p, 1, generator=g)[:, 0]
                toks[alive, t] = a[alive]
                length += alive.long()
                alive = alive & (a != END)
                prev = torch.where(a == END, prev, a)
        R = (toks[:, 0] == answer[q]).float()
        Rg = R.view(Q, G)
        adv = (Rg - Rg.mean(1, keepdim=True))
        if method in ("GRPO", "GRPO, summed"):
            adv = adv / (Rg.std(1, keepdim=True, unbiased=False) + 1e-4)   # population std, as in the worked example
        adv = adv.reshape(-1)
        # log-probs of the sampled tokens, recomputed with gradients
        prev = torch.cat([torch.full((Q * G, 1), V + 1), toks[:, :-1]], 1)
        for t in range(1, LMAX):                                      # an end token doesn't change "previous"
            prev[:, t] = torch.where(toks[:, t - 1] == END, prev[:, t - 1], toks[:, t - 1])
        lp = logits[q[:, None], torch.arange(LMAX)[None], prev].log_softmax(-1).gather(-1, toks[..., None])[..., 0]
        mask = torch.arange(LMAX)[None] < length[:, None]
        tok_loss = -(adv[:, None] * lp) * mask
        if method in ("GRPO", "Dr. GRPO, per answer"):
            loss = (tok_loss.sum(1) / length).mean()
        else:
            loss = tok_loss.sum(1).mean() / LMAX
        opt.zero_grad(); loss.backward(); opt.step()
        L = length.float()
        hist.append((R.mean().item(), L[R == 1].mean().item() if (R == 1).any() else np.nan,
                     L[R == 0].mean().item() if (R == 0).any() else np.nan))
    return np.array(hist)

for G in (4, 16):
    print(f"group size {G}: accuracy / mean length of right answers / of wrong answers, 5 seeds, at update")
    print(f"{'':22s}" + "".join(f"{u:>22d}" for u in (1, 50, 100, 400)))
    for method in ("GRPO", "GRPO, summed", "Dr. GRPO, per answer", "Dr. GRPO"):
        H = np.stack([run(method, G, seed) for seed in range(5)])
        at = lambda u: np.nanmean(H[:, max(0, u - 10):u], axis=(0, 1))
        print(f"{method:22s}" + "".join(f"{at(u)[0]:8.2f} {at(u)[1]:6.2f} {at(u)[2]:6.2f}" for u in (1, 50, 100, 400)))