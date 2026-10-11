import time
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP
import kernels
from lm import model, tok, DEVICE

# Kernel fusion, the first thing an engine build did, applied by hand to
# Qwen2.5-1.5B: every RMSNorm becomes one kernel instead of six operations,
# and each MLP's gate and up projections become one matmul followed by one
# fused SiLU-and-multiply. The RMSNorm kernel's version is the one tune.py
# found fastest for the row count. Measured before and after: operations
# per decoded token, time per decoded token, time to prefill 2,048 tokens,
# and how much the outputs change.
blog = tok(open("blog_train.txt").read()).input_ids
prompt = torch.tensor([blog[:2048]], device=DEVICE)

metal = 0                                  # our kernels, which PyTorch's dispatcher doesn't see

def fused_norm(self, x):
    global metal; metal += 1
    threads, vec = (512, 4) if x.numel() // x.shape[-1] <= 16 else (128, 4)
    return kernels.rmsnorm(x.contiguous(), self.weight, self.variance_epsilon, threads, vec)

def fused_mlp(self, x):
    global metal; metal += 1
    return self.down_proj(kernels.swiglu(F.linear(x, self.gate_up)))

ORIGINAL = Qwen2RMSNorm.forward, Qwen2MLP.forward
for m in model.modules():
    if isinstance(m, Qwen2MLP):
        m.gate_up = torch.cat([m.gate_proj.weight, m.up_proj.weight])       # [2 x 8960, 1536]

def use(fused):
    Qwen2RMSNorm.forward, Qwen2MLP.forward = (fused_norm, fused_mlp) if fused else ORIGINAL

class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.ops = 0
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.overloadpacket not in (torch.ops.aten.view, torch.ops.aten._unsafe_view):
            self.ops += 1
        return func(*args, **(kwargs or {}))

def ops_per_token():
    out = model(prompt[:, :16], use_cache=True)
    start = metal
    with Count() as c:
        model(prompt[:, 16:17], past_key_values=out.past_key_values, use_cache=True)
    return c.ops + metal - start

def timed(fn):
    torch.mps.synchronize(); t0 = time.perf_counter()
    out = fn(); torch.mps.synchronize()
    return time.perf_counter() - t0, out

gen = lambda: model.generate(prompt[:, :64], max_new_tokens=128, min_new_tokens=128, do_sample=False)
logits = lambda: model(prompt[:, :512]).logits[0].float()
r = {False: {"decode": [], "prefill": []}, True: {"decode": [], "prefill": []}}
with torch.no_grad():
    for fused in (False, True):
        use(fused); gen(); model(prompt)                          # warm up both
        r[fused]["ops"], r[fused]["logits"] = ops_per_token(), logits()
    for round in range(7):                                        # alternate, so drift hits both
        for fused in (False, True):
            use(fused)
            t, tokens = timed(gen)
            r[fused]["decode"].append(t / 128); r[fused]["tokens"] = tokens[0, 64:]
            r[fused]["prefill"].append(timed(lambda: model(prompt))[0])
for fused, label in ((False, "as written"), (True, "fused     ")):
    print(f"{label}: {r[fused]['ops']:4d} operations per decoded token; "
          f"{sorted(r[fused]['decode'])[3] * 1e3:5.1f} ms per decoded token; "
          f"prefill of 2,048 tokens {sorted(r[fused]['prefill'])[3] * 1e3:6.1f} ms")
ref, new = r[False]["logits"], r[True]["logits"]
top2 = ref.topk(2).values
print(f"next-token logits over 512 positions: largest difference {(new - ref).abs().max():.3f} "
      f"(logits span {ref.min():.0f} to {ref.max():.0f}); same top token at "
      f"{(new.argmax(-1) == ref.argmax(-1)).sum()} of 512, where the top two differ by "
      f"under 0.1 at {(top2[:, 0] - top2[:, 1] < 0.1).sum()}")
same = (r[False]["tokens"] == r[True]["tokens"]).cumprod(0).sum().item()
print(f"greedy output: the first {same} of 128 tokens are the same")
weights = sum(p.numel() * p.element_size() for n, p in model.named_parameters()
              if "embed_tokens" not in n or model.config.tie_word_embeddings)  # read once per token
step = sorted(r[True]["decode"])[3]
print(f"weights read per decoded token: {weights / 1e9:.2f} GB; at {step * 1e3:.1f} ms per token, "
      f"{weights / step / 1e9:.0f} GB/s")
