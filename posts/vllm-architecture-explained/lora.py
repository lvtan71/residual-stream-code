import time
import torch
from transformers import DynamicCache
from lm import model, tok, DEVICE

# Serving many LoRA fine-tunes from one base model (see "Parameter-Efficient
# Fine-Tuning"): every attention and MLP projection gets a stack of rank-16
# adapters, and each sequence in a batch names the adapter it uses. A decode
# step then adds, per projection, x A_i B_i for each sequence's own adapter i.
# Three ways:
#   shared:   one adapter for every sequence, two plain matrix multiplies;
#   gathered: each sequence's A_i and B_i copied out of the stack, then a
#             batched multiply (the simple way to mix adapters in a batch);
#   the same gathered path with one adapter for all, to separate the cost
#   of mixing adapters from the cost of the copying.
# Random adapter weights: only the cost is measured.
R, N_ADAPTERS = 16, 64
sync = torch.mps.synchronize if DEVICE == "mps" else torch.cuda.synchronize if DEVICE == "cuda" else (lambda: None)

class Multi(torch.nn.Module):
    which = None                                       # adapter per sequence, shared by every layer

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.A = torch.randn(N_ADAPTERS, base.in_features, R, dtype=base.weight.dtype, device=DEVICE) * 0.01
        self.B = torch.randn(N_ADAPTERS, R, base.out_features, dtype=base.weight.dtype, device=DEVICE) * 0.01

    def forward(self, x):                              # x: [batch, tokens, in]
        y = self.base(x)
        if Multi.which is None:
            return y
        if isinstance(Multi.which, str):               # "shared": adapter 0, no gathering
            return y + (x @ self.A[0]) @ self.B[0]
        i = Multi.which
        return y + torch.bmm(torch.bmm(x, self.A[i]), self.B[i])

for layer in model.model.layers:
    for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                          (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
        for name in names:
            setattr(parent, name, Multi(getattr(parent, name)))

adapter_bytes = sum(m.A[0].numel() + m.B[0].numel() for m in model.modules() if isinstance(m, Multi)) * 2
base_bytes = sum(p.numel() for n, p in model.named_parameters() if ".A" not in n and ".B" not in n) * 2
flops = sum(m.base.in_features * m.base.out_features for m in model.modules() if isinstance(m, Multi))
extra = sum(R * (m.base.in_features + m.base.out_features) for m in model.modules() if isinstance(m, Multi))
print(f"one rank-{R} adapter on all 7 projections of 28 layers: {adapter_bytes / 2**20:.0f} MiB, "
      f"against {base_bytes / 2**30:.2f} GiB for the base model; {extra / flops:.1%} more multiply-adds")

blog = tok(open("blog_train.txt").read()).input_ids

def timed(fn, trials=15):
    fn(); sync()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter(); fn(); sync()
        times.append(time.perf_counter() - t0)
    return sorted(times)[trials // 2]

@torch.no_grad()
def decode_ms(B, which, C=512):
    ids = torch.tensor([blog[i * C:(i + 1) * C] for i in range(B)], device=DEVICE)
    cache = DynamicCache()
    Multi.which = None
    model.model(ids, past_key_values=cache, use_cache=True)
    Multi.which = which
    def step():
        model(ids[:, -1:], past_key_values=cache, use_cache=True)
        cache.crop(-1)
    return timed(step) * 1e3

print("decode step, 512 cached tokens per sequence, ms (extra over the base model)")
print("     base    one adapter, shared   gathered, one adapter   gathered, a different adapter each")
for B in (16, 64):
    plain = decode_ms(B, None)
    shared = decode_ms(B, "shared")
    one = decode_ms(B, torch.zeros(B, dtype=torch.long, device=DEVICE))
    many = decode_ms(B, torch.arange(B, device=DEVICE) % N_ADAPTERS)
    print(f"{B:3d} {plain:6.1f}   {shared:7.1f} ({shared / plain - 1:+4.0%})       {one:7.1f} ({one / plain - 1:+4.0%})"
          f"          {many:7.1f} ({many / plain - 1:+4.0%})")
