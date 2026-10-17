import time
import torch
import torch.nn.functional as F
from gpt import GPT, Config
from train import final_split, tokenizer_for

# How much of the GPU a training step uses. Model FLOPs per token: 6 per
# parameter (6ND, from "Backpropagation, by Hand") plus attention's own work, 12 L T D (scores and
# weighted sums, forward and backward). Compared with the FP32 matmul peak
# measured on this machine in "How a GPU Runs Deep Learning": 2,878 GFLOP/s.
PEAK = 2878e9
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(DEVICE, lambda: None)   # the post ran on "mps"
for kind in ("char", "bpe"):
    V = tokenizer_for(kind, final_split["fit"])[0]
    c = Config(V=V)
    torch.manual_seed(0)
    m = GPT(c).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    B = 32
    x = torch.randint(0, V, (B, c.T + 1), device=DEVICE)
    n_params = sum(p.numel() for p in m.parameters())
    for i in range(25):
        if i == 5:
            sync(); t0 = time.perf_counter()
        loss = F.cross_entropy(m(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sync()
    step = (time.perf_counter() - t0) / 20
    tokens = B * c.T
    flops = tokens * (6 * n_params + 12 * c.L * c.T * c.D)
    print(f"{kind:4s}: {n_params:,} parameters, {step * 1e3:.0f} ms per step of {tokens:,} tokens, "
          f"{tokens / step:,.0f} tokens/s, {flops / step / 1e9:.0f} GFLOP/s, MFU {flops / step / PEAK:.1%}")