# DeepSeek-V4's KV cache per token, from its published configuration (the
# report, arXiv 2606.19348, plus the released config.json and FlashMLA's
# cache layout), against DeepSeek-V3.2 and a plain GQA baseline.
#
# Per layer, V4 keeps one compressed key/value entry (it is both: a single
# shared KV head) for every m tokens:
#   CSA layers: m = 4, plus an indexer key per entry for the sparse selection;
#   HCA layers: m = 128, no indexer.
# An entry is 512 numbers: 448 in FP8 plus the 64 RoPE dimensions in BF16,
# with 8 bytes of scales (7 one-byte UE8M0 scales + 1 pad): 584 bytes.
# The indexer key is 128 numbers in FP4 (64 bytes) plus scales, taken here
# as 4 bytes (one 1-byte scale per 32).
# Every layer also attends to a sliding window of the last 128 tokens; that
# cache is fixed in size, so per token of a 1M context it is negligible.
ENTRY, INDEXER, WINDOW, CONTEXT = 584, 64 + 4, 128, 2**20

def v4(csa_layers, hca_layers, total_layers):
    per_token = csa_layers * (ENTRY + INDEXER) / 4 + hca_layers * ENTRY / 128
    window = (total_layers + 1) * WINDOW * ENTRY / CONTEXT              # +1: the multi-token-prediction layer
    return per_token + window

pro, flash = v4(30, 31, 61), v4(21, 20, 43)
# V3.2: MLA on all 61 layers, one 656-byte entry per token per layer
# (512 FP8 + 16 bytes of scales + 64 BF16 RoPE dims), plus an FP8 indexer key
# of 128 numbers and a 4-byte scale.
v32 = 61 * (656 + 132)
# GQA baseline: 8 key-value heads of 128 dimensions, keys and values, BF16.
gqa = lambda layers: layers * 2 * 8 * 128 * 2

for name, b, layers in (("DeepSeek-V4-Pro", pro, 61), ("DeepSeek-V4-Flash", flash, 43)):
    print(f"{name:18s} {b:7,.0f} bytes per token, {b * CONTEXT / 2**30:5.1f} GiB at 1M tokens; "
          f"{b / v32:5.1%} of V3.2's, {b / gqa(layers):4.1%} of a {layers}-layer GQA-8 model's")
print(f"{'DeepSeek-V3.2':18s} {v32:7,.0f} bytes per token, {v32 * CONTEXT / 2**30:5.1f} GiB at 1M tokens")
print(f"{'GQA-8, 61 layers':18s} {gqa(61):7,.0f} bytes per token, {gqa(61) * CONTEXT / 2**30:5.1f} GiB at 1M tokens")
csa_share = 30 * (ENTRY + INDEXER) / 4 / pro
print(f"V4-Pro: {csa_share:.0%} of its cache is in the 30 CSA layers; the 31 HCA layers hold "
      f"{31 * ENTRY / 128 / pro:.0%}")