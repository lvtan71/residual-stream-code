# What thinking tokens cost, for one request on two real configurations
# (from their config.json files):
#   Qwen2.5-32B: 64 layers, 8 key/value heads of 128 (GQA, "The KV Cache"), BF16;
#   DeepSeek-R1 (671B, 37B active): 61 layers of multi-head latent
#   attention, caching a 512-number latent plus a 64-number RoPE key per
#   token per layer ("Anatomy of a Frontier Model"), BF16.
# Decode time, batch 1, bandwidth-bound ("Small and Specialized Models"): each token reads the
# active weights plus the whole KV cache so far, at one H100's 3.35 TB/s.
# Qwen2.5-32B fits on one H100; DeepSeek-R1 needs many GPUs to hold its
# 671B parameters, so its row only illustrates the reading: its active
# weights and cache at one GPU's bandwidth, without the communication
# between GPUs that a real deployment adds ("Mixture of Experts").
H100 = 3.35e12
models = {
    "Qwen2.5-32B": dict(kv=2 * 64 * 8 * 128 * 2, weights=32.5e9 * 2),
    "DeepSeek-R1": dict(kv=61 * (512 + 64) * 2, weights=37e9 * 1),   # FP8 weights, the format it was trained in
}
print(f"{'':12s} {'KV per token':>13s}   {'thinking tokens':>15s} {'KV cache':>9s} {'decode time':>12s} {'avg tokens/s':>13s}")
for name, m in models.items():
    for n in (1_000, 10_000, 50_000):
        # time = sum over tokens of (weights + KV so far) / bandwidth
        seconds = (n * m["weights"] + m["kv"] * n * (n - 1) / 2) / H100
        print(f"{name:12s} {m['kv'] / 1024:9.1f} KiB   {n:15,d} {m['kv'] * n / 2**30:7.2f} GiB {seconds:10.1f} s {n / seconds:13.0f}")