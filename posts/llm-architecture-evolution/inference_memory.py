# What the attention substitutions save: the KV cache per generated token
# ("The KV Cache"), in 16-bit numbers, for published configurations, and the
# attention weights per layer. MHA keeps a key and a value per head; GQA one
# per group of query heads; MLA a compressed vector per token plus a small
# rotated key (DeepSeek-AI, 2024).
def kv_bytes(layers, per_layer_elements, bytes_per=2):
    return layers * per_layer_elements * bytes_per

configs = [
    # name, layers, model width, query heads, key/value heads, head size
    ("GPT-3 175B (MHA)", 96, 12288, 96, 96, 128),
    ("Llama 2 70B (GQA, 8 groups)", 80, 8192, 64, 8, 128),
    ("Llama 3 8B (GQA, 8 groups)", 32, 4096, 32, 8, 128),
]
for name, L, D, nq, nkv, h in configs:
    cache = kv_bytes(L, 2 * nkv * h)
    attn = D * nq * h * 2 + D * nkv * h * 2                     # Q and output, then K and V
    print(f"{name:30s} KV cache {cache / 1024:7.0f} KiB per token; attention weights per layer "
          f"{attn / 1e6:6.1f}M (with full MHA: {4 * D * nq * h / 1e6:6.1f}M)")

# DeepSeek-V2: 60 layers, 128 heads of 128, KV compression 512, rotated key 64.
L, nh, dh, dc, dr = 60, 128, 128, 512, 64
mla, mha = kv_bytes(L, dc + dr), kv_bytes(L, 2 * nh * dh)
print(f"{'DeepSeek-V2 (MLA)':30s} KV cache {mla / 1024:7.0f} KiB per token; the same heads with MHA: "
      f"{mha / 1024:,.0f} KiB, {mha / mla:.0f} times as much")

# Mixture of experts: parameters held vs parameters used per token.
for name, total, active in (("Mixtral 8x7B", 47e9, 13e9), ("DeepSeek-V2", 236e9, 21e9)):
    print(f"{name:30s} {total / 1e9:5.1f}B parameters, {active / 1e9:4.1f}B used per token "
          f"({active / total:.0%}): per-token FLOPs of a {active / 1e9:.0f}B model, memory of a {total / 1e9:.0f}B one")
