# What a context of n tokens holds in a model's KV cache: 2 (K and V) x layers
# x KV heads x head dim x 2 bytes per token (16-bit), the formula from
# The KV Cache. Claude's architecture is not public; these are open models
# of known shape, as stand-ins.
models = {"Llama 2 7B (32 layers, 32 KV heads of 128)": (32, 32, 128),
          "Llama 2 70B (80 layers, 8 KV heads of 128)": (80, 8, 128),
          "Qwen2.5-1.5B (28 layers, 2 KV heads of 128)": (28, 2, 128)}
contexts = {"this post's baseline peak": 90_421, "the cookbook's baseline peak": 335_279}
for name, (layers, heads, dim) in models.items():
    per_token = 2 * layers * heads * dim * 2
    print(f"{name}: {per_token / 1024:,.0f} KiB per token")
    for label, n in contexts.items():
        print(f"   {label}, {n:,} tokens: {per_token * n / 2**30:6.1f} GiB")
