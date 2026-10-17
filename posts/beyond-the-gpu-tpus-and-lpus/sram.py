import math
# Holding a 70B-parameter model, and the fastest one stream of tokens could
# go, on four kinds of chip (vendor specs). Decoding one token at batch 1
# reads every weight once, so a bound on its time is weight bytes divided by
# the memory bandwidth the weights sit behind, or FLOPs divided by compute,
# whichever is larger. Bounds, not measurements: they ignore the links
# between chips, the KV cache and any software overhead.
P = 70e9
chips = {   # name: (memory per chip in bytes, bandwidth to it in bytes/s, dense FLOP/s at 16-bit)
    "GroqChip (SRAM)":  (230e6,          80e12,   188e12),
    "H100 SXM (HBM)":   (80e9,           3.35e12, 989e12),
    "TPU v6e (HBM)":    (32e9,           1.638e12, 918e12),
    "TPU7x (HBM)":      (192 * 2 ** 30,  7.38e12, 2307e12),
}
for bytes_per_param, label in ((2, "16-bit weights"), (1, "8-bit weights")):
    W = P * bytes_per_param
    print(f"70B model, {label}: {W / 1e9:.0f} GB")
    print(f"  {'chip':18s} {'chips to hold it':>16s} {'tokens/s bound':>15s} {'on 8 chips':>11s}")
    for name, (mem, bw, flops) in chips.items():
        n = math.ceil(W / mem)
        bound = lambda k: 1 / max(W / (k * bw), 2 * P / (k * flops))
        if name.startswith("Groq"):
            # Layers are spread across chips and run one after another for a
            # given token: each chip reads only its own share, but in turn, so
            # one stream's time is all the bytes over one chip's bandwidth,
            # however many chips there are.
            print(f"  {name:18s} {n:16d} {1 / max(W / bw, 2 * P / flops):15.0f} {'-':>11s}")
        else:
            # Tensor parallelism: k chips each read their share at the same time.
            print(f"  {name:18s} {n:16d} {bound(n):15.0f} {bound(max(n, 8)):11.0f}")
    print()