import math
import torch
import torch.nn.functional as F
from gpt import GPT, Config
from train import dev_split, tokenizer_for

# Is the collapse a matter of settings? The model without residual
# connections at lower learning rates and without dropout, 1,000 steps each:
# train loss, and whether its output still depends on its input.
torch.set_num_threads(4)
V, encode, _ = tokenizer_for("char", dev_split["fit"])
fit = encode(dev_split["fit"])
for lr, dropout in ((1e-3, 0.0), (3e-4, 0.2), (1e-4, 0.2)):
    torch.manual_seed(0); g = torch.Generator().manual_seed(0)
    m = GPT(Config(V=V, residual=False, dropout=dropout))
    opt = torch.optim.AdamW(m.parameters(), lr, betas=(0.9, 0.95))
    for step in range(1, 1001):
        for gr in opt.param_groups:
            gr["lr"] = lr * min(1, step / 200)
        i = torch.randint(0, len(fit) - 257, (16,), generator=g)
        idx = i[:, None] + torch.arange(257)
        x, y = fit[idx[:, :-1]], fit[idx[:, 1:]]
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    with torch.no_grad():
        m.eval()
        a, b = m(x[:1])[0, -1].softmax(-1), m(x[1:2])[0, -1].softmax(-1)
    print(f"lr {lr:.0e}, dropout {dropout}: train loss {loss.item() / math.log(2):.3f} bits after 1,000 steps; "
          f"two contexts' predictions differ by {0.5 * (a - b).abs().sum():.3f}", flush=True)
