import numpy as np
import torch
# Is DDIM's collapse at 5 steps about the number of steps, or where they
# fall? diff2d.py's trained models, sampled with three choices of the
# timesteps to visit. diff2d.py's own choice, evenly spaced from 999 down
# to 0, wastes its last step at t = 0 (almost no noise left) and leaves
# one jump from t = 250 to clean data.
src = open("diff2d.py").read().split("nets = [train(s)")[0]
ns = {}; exec(src.replace("print(", "(lambda *a, **k: None)("), ns)
nets = []
for sd in torch.load("nets.pt"):
    m = ns["Eps"](); m.load_state_dict(sd); nets.append(m)
abar, score = ns["abar"], ns["score"]

@torch.no_grad()
def ddim_at(net, n, ts):
    """diff2d.py's ddim(), visiting the given timesteps (descending)."""
    x, c = torch.randn(n, 2), torch.full((n,), 8)
    for i, t in enumerate(ts):
        e = ns["guided"](net, x, t, c, 1.0)
        x0 = (x - (1 - abar[t]).sqrt() * e) / abar[t].sqrt()
        a_next = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * e
    return x

spacings = {
    "even, ending at t = 0 (diff2d.py)": lambda k: torch.linspace(999, 0, k).round().long().tolist(),
    "even, ending at t = 999/k": lambda k: [round(999 - i * 1000 / k) for i in range(k)],
    "quadratic (DDIM paper)": lambda k: sorted({round(999 * (1 - i / k) ** 2) for i in range(k)}, reverse=True),
}
print("unconditional, 2,000 samples per seed: share near a mode (modes covered)")
for k in (5, 10, 20):
    for name, f in spacings.items():
        ts = f(k)
        torch.manual_seed(100)
        res = [score(ddim_at(m, 2000, ts)) for m in nets]
        print(f"{k:3d} steps, {name:34s} {'  '.join(f'{r[1]:4.0%} ({r[0]})' for r in res)}   t = {ts}"
              if k == 5 else
              f"{k:3d} steps, {name:34s} {'  '.join(f'{r[1]:4.0%} ({r[0]})' for r in res)}", flush=True)