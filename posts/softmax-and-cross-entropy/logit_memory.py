# Memory for the logits of one training batch, at Llama-3-8B's vocabulary.
V, D = 128_256, 4096
for B, T in [(1, 8192), (8, 8192)]:
    n = B * T                                            # token positions
    logits = n * V * 4                                   # FP32 logits
    print(f"B={B}, T={T}: {n:,} positions x {V:,} = {n * V / 1e9:.2f} B logits; "
          f"FP32 {logits / 2**30:.1f} GiB, and the same again for their gradient")
print(f"the hidden states that produce them, BF16: {8 * 8192 * D * 2 / 2**30:.2f} GiB")
print(f"with activation checkpointing, the saved input of each of 32 layers: "
      f"{32 * 8 * 8192 * D * 2 / 2**30:.0f} GiB")