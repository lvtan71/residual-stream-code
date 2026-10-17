import math
import torch
import torch.nn.functional as F
from gpt import GPT, Config
from train import dev_split, tokenizer_for

# Why does the model without residual connections stop at the unigram level?
# Train loss, and the gradient reaching each layer at the first steps.
torch.set_num_threads(4)
V, encode, _ = tokenizer_for("char", dev_split["fit"])
fit = encode(dev_split["fit"])
counts = torch.bincount(fit, minlength=V).float()
p = counts / counts.sum()
print(f"unigram entropy of the training text: {-(p[p > 0] * p[p > 0].log2()).sum():.3f} bits per character")
for residual in (True, False):
    torch.manual_seed(0); g = torch.Generator().manual_seed(0)
    m = GPT(Config(V=V, residual=residual, dropout=0.2))
    opt = torch.optim.AdamW(m.parameters(), 1e-3, betas=(0.9, 0.95))
    for step in range(1, 401):
        for gr in opt.param_groups:
            gr["lr"] = 1e-3 * min(1, step / 200)
        i = torch.randint(0, len(fit) - 257, (16,), generator=g)
        idx = i[:, None] + torch.arange(257)
        x, y = fit[idx[:, :-1]], fit[idx[:, 1:]]
        logits = m(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        if step in (1, 400):
            norms = [m.blocks[l].qkv.weight.grad.norm().item() for l in range(4)]
            print(f"residual={residual!s:5} step {step:3d}: train loss {loss.item() / math.log(2):.3f} bits; "
                  f"gradient norm at the attention weights of layers 1-4: "
                  + " ".join(f"{n:.1e}" for n in norms))
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    with torch.no_grad():                       # does the output depend on the input at all?
        m.eval()
        a, b = m(x[:1])[0, -1].softmax(-1), m(x[1:2])[0, -1].softmax(-1)
        print(f"   output distributions for two different contexts differ by {0.5 * (a - b).abs().sum():.3f} (total variation)")