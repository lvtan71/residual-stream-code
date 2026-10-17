# The KV cache's budget, worked for Llama-2-13B (40 layers, 40 heads of 128,
# full multi-head attention): bytes per token of context, and for one
# 32,768-token sequence, under each lever and their combinations. Group
# scales for quantized caches are counted (a 16-bit scale and zero-point
# per 32 numbers adds 1 bit per number).
L, H, D, CTX = 40, 40, 128, 32_768
full = 2 * L * H * D * 2                                     # K and V, BF16
rows = [("BF16, every token (baseline)", 16, 1.0, H),
        ("if it had 8 KV heads (GQA, post 18)", 16, 1.0, 8),
        ("4-bit K and V", 4 + 1, 1.0, H),
        ("2-bit K and V (KIVI)", 2 + 1, 1.0, H),
        ("keep a quarter of the tokens", 16, 0.25, H),
        ("GQA-8 + 4-bit + a quarter of the tokens", 4 + 1, 0.25, 8)]
print(f"{'':42s} {'bytes per token':>16s} {'one 32k sequence':>17s} {'vs baseline':>12s}")
for name, bits, frac, heads in rows:
    b = 2 * L * heads * D * bits / 8 * frac
    print(f"{name:42s} {b:14,.0f} B {b * CTX / 2**30:13.2f} GiB {b / full:11.1%}")