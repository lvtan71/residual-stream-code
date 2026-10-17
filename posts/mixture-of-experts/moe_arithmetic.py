# Mixture-of-experts arithmetic from published configurations.
# Each expert is a SwiGLU MLP: 3 * D * F weights (gate, up, down).
def expert(D, F):
    return 3 * D * F

# Mixtral 8x7B: 32 layers, D = 4,096, 8 experts of F = 14,336, top-2.
D, L, F, E, k = 4096, 32, 14336, 8, 2
print(f"Mixtral 8x7B: one expert {expert(D, F) / 1e6:.0f}M weights; all experts, all layers "
      f"{expert(D, F) * E * L / 1e9:.1f}B; used per token {expert(D, F) * k * L / 1e9:.1f}B")

# DeepSeek-V3: 61 layers, the first 3 dense (F = 18,432); the other 58 have
# 256 routed experts of F = 2,048, top-8, plus 1 shared expert every token uses.
D, L, dense, F_dense, E, F, k, shared = 7168, 61, 3, 18432, 256, 2048, 8, 1
routed_all = expert(D, F) * E * (L - dense)
used = (expert(D, F) * (k + shared)) * (L - dense) + expert(D, F_dense) * dense
print(f"DeepSeek-V3: one routed expert {expert(D, F) / 1e6:.0f}M weights; routed experts, all layers "
      f"{routed_all / 1e9:.0f}B of the reported 671B; MLP weights used per token {used / 1e9:.1f}B of the reported 37B active")

# Expert parallelism: experts live on different GPUs, so every token's
# hidden vector travels to its k experts and the results travel back.
# DeepSeek-V3 dispatches in FP8 (1 byte) and combines in BF16 (2 bytes).
per_layer = k * D * (1 + 2)
print(f"all-to-all per token: {per_layer / 1024:.0f} KiB per MoE layer, {per_layer * (L - dense) / 2**20:.1f} MiB "
      f"across the {L - dense} MoE layers (at most; tokens whose experts share a GPU send less)")

# Decoding: a batch of B tokens, each picking k of E experts. How many
# experts does a step touch, and how many tokens does each one get? An
# expert that processes t tokens does 2 * t FLOPs per weight it reads.
for B in (1, 8, 64, 512, 4096):
    touched = E * (1 - (1 - k / E) ** B)
    print(f"batch {B:>5,}: about {touched:5.1f} of {E} experts touched per layer, "
          f"{B * k / touched:6.1f} tokens per touched expert")
