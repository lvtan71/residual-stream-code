# The cost sketch of "AI Engineering, from the Inside", redone with what the assistant measured.
# There: an 8B model (32 layers, 8 KV heads of 128) in BF16 on one H100
# (989 TFLOPS, 3.35 TB/s), with a guessed 2,000-token prompt and a
# 300-token answer. Here: the scorecard's averages, 699 tokens in and 52
# out. Prefill costs 2 x parameters FLOPs per token; each decode step reads
# all the weights, plus every sequence's KV cache when batched.
P, FLOPS, BW = 8e9, 989e12, 3.35e12
weights = 2 * P                                           # bytes in BF16
kv_per_token = 32 * 8 * 128 * 2 * 2                       # layers x heads x dim x (K, V) x bytes = 128 KiB
for label, n_in, n_out in (("series map's guess", 2000, 300), ("measured here", 699, 52)):
    prefill = 2 * P * n_in / FLOPS
    single = prefill + n_out * weights / BW
    kv = (n_in + n_out) * kv_per_token
    step64 = (weights + 64 * kv) / BW
    batched = prefill + n_out * step64 / 64
    print(f"{label:20s} {n_in:5d} in, {n_out:3d} out: batch 1 {single:5.2f} GPU-s per answer; "
          f"batch 64 {batched * 1e3:4.0f} ms per answer")