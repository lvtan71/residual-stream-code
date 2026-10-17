import numpy as np
import torch
import torch.nn.functional as F

# DPO on "RLHF"'s exact toy: 512 responses, a hidden gold reward, 300
# comparisons drawn from it through Bradley-Terry ("random" world, seed 0).
exec(open("rlhf_toy.py").read().split("betas = [")[0])     # "RLHF"'s world(), X, Y and sigmoid
torch.set_default_dtype(torch.float64)
rng, gold, ref = world("random", 0)
n = 300
a, b = rng.choice(len(Y), n, p=ref), rng.choice(len(Y), n, p=ref)
p_true = sigmoid(gold[a] - gold[b])
prefer_a = rng.random(n) < p_true
win, lose = torch.tensor(np.where(prefer_a, a, b)), torch.tensor(np.where(prefer_a, b, a))
p_win = torch.tensor(np.where(prefer_a, p_true, 1 - p_true))         # the true preference probability
log_ref = torch.tensor(np.log(ref))
Xt = torch.tensor(X)

def policy(params, kind):
    """log pi over all 512 responses. "features": log pi_ref + X theta, the
    family "RLHF"'s reward model lives in; "table": one free logit per
    response."""
    z = log_ref + (Xt @ params if kind == "features" else params)
    return z - torch.logsumexp(z, 0)

def loss_fn(params, kind, loss_kind, beta):
    lp = policy(params, kind)
    h = (lp[win] - log_ref[win]) - (lp[lose] - log_ref[lose])           # log-ratio margin
    if loss_kind == "DPO":
        return -F.logsigmoid(beta * h).mean()
    if loss_kind == "DPO, soft labels":                                 # labels = true preference probability
        return -(p_win * F.logsigmoid(beta * h) + (1 - p_win) * F.logsigmoid(-beta * h)).mean()
    return ((h - 1 / (2 * beta)) ** 2).mean()                           # IPO

# 1. In the feature family, the DPO loss in theta is the Bradley-Terry loss
#    of "RLHF"'s reward model in beta * theta: the same convex problem.
#    Both solved to convergence (L-BFGS), then compared as policies.
BETA = 0.5
def solve(f, dim):
    x = torch.zeros(dim, requires_grad=True)
    opt = torch.optim.LBFGS([x], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
    for _ in range(20):
        opt.step(lambda: (opt.zero_grad(), (l := f(x)).backward(), l)[2])
    return x.detach()
D = Xt[win] - Xt[lose]
theta = solve(lambda t: -F.logsigmoid(D @ t).mean(), X.shape[1])          # reward model, unregularized
rlhf = (log_ref + Xt @ theta / BETA).softmax(0).numpy()
dpo = policy(solve(lambda t: loss_fn(t, "features", "DPO", BETA), X.shape[1]), "features").exp().numpy()
print(f"feature family, beta = {BETA}: RLHF policy vs DPO policy, largest difference in any response's "
      f"probability: {np.abs(rlhf - dpo).max():.0e}; KL from ref {np.sum(dpo * np.log(dpo / ref)):.2f}, "
      f"gold gain {dpo @ gold - ref @ gold:.2f}")

# 2. One free logit per response.
w_set, l_set = set(win.tolist()), set(lose.tolist())
only_won = torch.tensor(sorted(w_set - l_set))
seen = torch.tensor(sorted(w_set | l_set))
def run(loss_kind, beta, opt_name="Adam", steps=30000):
    params = torch.zeros(len(Y), requires_grad=True)
    opt = torch.optim.Adam([params], lr=0.05) if opt_name == "Adam" else torch.optim.SGD([params], lr=1.0)
    for step in range(1, steps + 1):
        loss = loss_fn(params, "table", loss_kind, beta)
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (1000, steps):
            with torch.no_grad():
                lp = policy(params, "table"); pi = lp.exp()
                unseen = 1 - pi[seen].sum().item()
                print(f"{loss_kind:16s} {beta:4g} {opt_name:4s} {step:6d} {float(pi @ (lp - log_ref)):6.2f} "
                      f"{float(pi.numpy() @ gold - ref @ gold):6.2f} {unseen:7.2f} {pi[only_won].sum().item():9.2f} "
                      f"{pi.topk(2).values.sum().item():6.2f} {float((lp - log_ref)[win].mean()):9.1f}")
print(f"\none free logit per response; the 300 comparisons mention {len(seen)} of 512 responses, "
      f"{len(only_won)} of which only ever won")
print(f"{'loss':16s} {'beta':>4s} {'opt':4s} {'steps':>6s} {'KL':>6s} {'gold':>6s} {'unseen':>7s} {'only won':>9s} "
      f"{'top 2':>6s} {'chosen log ratio':>9s}")
for args in (("DPO", 0.5), ("DPO", 0.5, "SGD"), ("DPO, soft labels", 0.5), ("IPO", 0.5), ("IPO", 0.1)):
    run(*args)
