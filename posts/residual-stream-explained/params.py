import torch

# Two MLP sublayers of the same size. The 2017 one expands D to 4D, applies
# a nonlinearity, and projects back. SwiGLU (Llama and most current models)
# has three matrices: a gate, an "up" projection, and the way back down, with
# a hidden width of about 8/3 D so it costs the same.
class SwiGLU(torch.nn.Module):
    def __init__(self, D, F):
        super().__init__()
        self.gate = torch.nn.Linear(D, F, bias=False)
        self.up = torch.nn.Linear(D, F, bias=False)
        self.down = torch.nn.Linear(F, D, bias=False)
    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

count = lambda m: sum(p.numel() for p in m.parameters())
D = 4096
with torch.device("meta"):                                  # shapes only, no memory
    gelu_mlp = torch.nn.Sequential(torch.nn.Linear(D, 4 * D, bias=False), torch.nn.GELU(),
                                   torch.nn.Linear(4 * D, D, bias=False))
    swiglu = SwiGLU(D, 11008)                               # Llama 1 7B: 8/3 * 4096, rounded up to a multiple of 256
    attention = torch.nn.ModuleList([torch.nn.Linear(D, D, bias=False) for _ in range(4)])  # Q, K, V, output
print(f"D = {D}: GELU MLP {count(gelu_mlp) / 1e6:.1f}M weights (8 D^2 = {8 * D * D / 1e6:.1f}M), "
      f"SwiGLU {count(swiglu) / 1e6:.1f}M, attention {count(attention) / 1e6:.1f}M (4 D^2)")

# Forward FLOPs per token for one block: 2 per weight for every matrix, plus
# attention's own work: for each earlier token, D multiply-adds (2D FLOPs)
# for its score and 2D more for adding in its value. With the causal mask a
# token sees, on average over a sequence of length T, T/2 earlier tokens.
for T in (2048, 8192, 32768, 131072):
    weights = 2 * (count(swiglu) + count(attention))
    mixing = 4 * D * T / 2
    print(f"context {T:>7,}: MLP {2 * count(swiglu) / (weights + mixing):.0%} of the block's FLOPs, "
          f"attention's projections {2 * count(attention) / (weights + mixing):.0%}, "
          f"scores and weighted sums {mixing / (weights + mixing):.0%}")

# A newer layout: Llama 3 8B widens the MLP to 14,336 and shares each key and
# value head among 4 query heads (grouped-query attention), so K and V
# project to 1,024 dimensions instead of 4,096.
with torch.device("meta"):
    mlp3 = SwiGLU(D, 14336)
    attn3 = torch.nn.ModuleList([torch.nn.Linear(D, n, bias=False) for n in (D, 1024, 1024, D)])
print(f"Llama 3 8B block: MLP {count(mlp3) / 1e6:.1f}M, attention {count(attn3) / 1e6:.1f}M, "
      f"MLP share {count(mlp3) / (count(mlp3) + count(attn3)):.0%}")

# And a real model's weights by role: Qwen2.5-1.5B-Instruct, built from its
# configuration on the meta device (shapes only, nothing downloaded but the
# config). Its embedding and unembedding are one shared (tied) matrix.
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
with torch.device("meta"):
    qwen = AutoModelForCausalLM.from_config(cfg)
roles = {"embedding (tied)": 0, "attention": 0, "MLP": 0, "norms": 0}
for name, p in qwen.named_parameters():
    role = ("embedding (tied)" if "embed" in name else "attention" if "self_attn" in name
            else "MLP" if "mlp" in name else "norms")
    roles[role] += p.numel()
total = sum(roles.values())
print(f"Qwen2.5-1.5B-Instruct: {total / 1e9:.2f}B weights, D = {cfg.hidden_size}, MLP width {cfg.intermediate_size}, "
      f"{cfg.num_hidden_layers} blocks")
for role, n in roles.items():
    print(f"  {role:17s} {n / 1e6:7.1f}M  {n / total:4.0%}")
