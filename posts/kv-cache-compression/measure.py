import math
import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from huggingface_hub import snapshot_download

# Shrinking the KV cache, measured on Qwen2.5-0.5B (base) on the CPU: the
# loss on held-out text from this blog when the model may only attend to
# part of its cache (eviction), or when the cached keys and values are
# rounded to fewer bits (quantization). Eight windows of 2,048 tokens; the
# last 1,024 tokens of each are scored, so each has 1,024 or more tokens of
# context. Loss in bits per token; changes are against the full cache, with
# a standard error over the 8,192 scored tokens.
P = snapshot_download("Qwen/Qwen2.5-0.5B",
                      revision="060db6499f32faf8b98477b0a26969ef7d8b9987")
tok = AutoTokenizer.from_pretrained(P)
ids = tok(open("blog_val.txt").read(), return_tensors="pt").input_ids[0]
T = 2048
seqs = [ids[i * T:(i + 1) * T] for i in range(len(ids) // T)]

def round_to(x, bits, per, group):
    """Round x (..., tokens, channels) to `bits`, with a scale and zero-point
    for each group: per="token", groups of `group` channels of one token;
    per="channel", one channel over groups of `group` tokens."""
    if per == "token":
        g = x.reshape(*x.shape[:-1], -1, group)
        lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    else:
        g = x.reshape(*x.shape[:-2], x.shape[-2] // group, group, x.shape[-1])
        lo, hi = g.amin(-2, keepdim=True), g.amax(-2, keepdim=True)
    s = (hi - lo).clamp(min=1e-8) / (2 ** bits - 1)
    return (((g - lo) / s).round() * s + lo).reshape(x.shape)

# The cache as the attention reads it: keys after RoPE, as real caches store
# them, rounded by CACHE's settings; the last R tokens before each query
# kept in full precision (KIVI's residual window).
CACHE = {}
def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    kq = round_to(key, *CACHE["k"]) if "k" in CACHE else key
    vq = round_to(value, *CACHE["v"]) if "v" in CACHE else value
    K, V, Kq, Vq = (repeat_kv(t, module.num_key_value_groups) for t in (key, value, kq, vq))
    scores = query @ Kq.transpose(2, 3) * scaling
    R = CACHE.get("R", 0)
    if R:
        near = (torch.arange(T)[:, None] - torch.arange(T)[None]) < R
        scores = torch.where(near, query @ K.transpose(2, 3) * scaling, scores)
    w = torch.softmax(scores + attention_mask, -1)
    out = (w * near) @ V + (w * ~near) @ Vq if R else w @ Vq
    return out.transpose(1, 2).contiguous(), None
AttentionInterface.register("kv_experiment", attention)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float32, attn_implementation="kv_experiment").eval()

# A control: keys rounded per token *before* RoPE, with the key projection's
# bias taken out first and added back after.
DEBIASED = {}
def hook(mod, inp, out):
    if not DEBIASED:
        return out
    x = out[0] - mod.bias
    x = round_to(x.reshape(T, -1, 64).transpose(0, 1), DEBIASED["bits"], "token", 64).transpose(0, 1).reshape(T, -1)
    return (x + mod.bias)[None]
for layer in m.model.layers:
    layer.self_attn.k_proj.register_forward_hook(hook)

def mask(keep):
    M = torch.full((T, T), float("-inf"))
    for i in range(T):
        M[i, keep(i)] = 0
    return M[None, None]
causal = mask(lambda i: slice(0, i + 1))

@torch.no_grad()
def losses(attention_mask=causal):
    out = []
    for s in seqs:
        logits = m(s[None], attention_mask=attention_mask).logits[0]
        out.append(F.cross_entropy(logits[1023:-1], s[1024:], reduction="none") / math.log(2))
    return torch.cat(out)

base = losses()
def report(name, l):
    d = l - base
    print(f"  {name:44s} {l.mean():.3f}  (+{d.mean():.3f} ± {d.std() / math.sqrt(len(d)):.3f})", flush=True)
print(f"full cache: {base.mean():.3f} bits per token, {len(seqs)} windows")
print("eviction: keep")
for budget in (512, 256, 128):
    report(f"the last {budget} tokens", losses(mask(lambda i: slice(max(0, i - budget + 1), i + 1))))
    report(f"4 sink tokens + the last {budget - 4}",
           losses(mask(lambda i: sorted(set(range(min(4, i + 1))) | set(range(max(0, i - budget + 5), i + 1))))))
print("quantization of the cache (keys after RoPE):")
for name, cfg in (("8-bit, per token", dict(k=(8, "token", 64), v=(8, "token", 64))),
                  ("4-bit, per token", dict(k=(4, "token", 64), v=(4, "token", 64))),
                  ("4-bit, only values, per token", dict(v=(4, "token", 64))),
                  ("4-bit, only keys, per token", dict(k=(4, "token", 64))),
                  ("4-bit, keys per channel", dict(k=(4, "channel", 32), v=(4, "token", 64))),
                  ("2-bit, per token", dict(k=(2, "token", 64), v=(2, "token", 64))),
                  ("2-bit, keys per channel", dict(k=(2, "channel", 32), v=(2, "token", 64))),
                  ("2-bit, keys per channel, last 128 exact (KIVI)", dict(k=(2, "channel", 32), v=(2, "token", 64), R=128))):
    CACHE.update(cfg); report(name, losses()); CACHE.clear()
print("control: keys per token with the key bias taken out first (before RoPE)")
for bits in (4, 2):
    DEBIASED["bits"] = bits; report(f"{bits}-bit keys, per token, bias removed", losses()); DEBIASED.clear()

# The bias: a constant offset per channel, the same for every token.
seen = {}
hooks = [m.model.layers[l].self_attn.k_proj.register_forward_hook(lambda mod, i, o, l=l: seen.__setitem__(l, o[0]))
         for l in (0, 12)]
with torch.no_grad():
    m(seqs[0][None], attention_mask=causal)
for h in hooks:
    h.remove()
for l, k in seen.items():
    b = m.model.layers[l].self_attn.k_proj.bias
    span = lambda x: (x.reshape(T, -1, 64).amax(-1) - x.reshape(T, -1, 64).amin(-1)).mean()
    print(f"layer {l:2d} keys: largest bias {b.abs().max():5.1f}; a token's 64 channels span {span(k):5.1f} on average, "
          f"{span(k - b):4.1f} without the bias; a channel varies from token to token by {(k - k.mean(0)).abs().mean():.2f}")