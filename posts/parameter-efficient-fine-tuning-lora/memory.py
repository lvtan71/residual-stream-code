# LoRA's parameter and memory arithmetic for Llama 3.1 8B (its config.json:
# hidden 4096, MLP 14336, 32 layers, 32 query heads and 8 key/value heads of
# 128, vocabulary 128256). LoRA at rank r on a d_out x d_in matrix trains
# r * (d_in + d_out) numbers instead of d_in * d_out.
D, F, L, KV, V = 4096, 14336, 32, 8 * 128, 128256
mats = {"q": (D, D), "k": (D, KV), "v": (D, KV), "o": (D, D), "gate": (D, F), "up": (D, F), "down": (F, D)}
per_layer = sum(i * o for i, o in mats.values())
total = L * per_layer + 2 * V * D + (2 * L + 1) * D       # + input and output embeddings (untied), norm gains
print(f"Llama 3.1 8B: {total / 1e9:.2f}B parameters, {L * per_layer / 1e9:.2f}B of them in the 7 matrices of each block")
for r in (1, 8, 16, 64):
    lora = L * sum(r * (i + o) for i, o in mats.values())
    print(f"  LoRA r = {r:2d} on all 7: {lora / 1e6:6.1f}M trainable, {lora / total:.2%} of the model")
r16 = L * sum(16 * (i + o) for i, o in mats.values())
GB = 1e9
print("\nMemory for training, weights + gradients + AdamW's two moments (activations not counted):")
print(f"  full fine-tune, mixed precision (16 bytes per parameter, post 4.1): {16 * total / GB:6.1f} GB")
print(f"  LoRA r = 16: 16-bit frozen weights {2 * total / GB:.1f} GB + 16 bytes per LoRA parameter "
      f"{16 * r16 / GB:.2f} GB = {(2 * total + 16 * r16) / GB:6.1f} GB")
q4 = 0.5 * L * per_layer + 2 * (total - L * per_layer)          # 4-bit block matrices, 16-bit embeddings and norms
print(f"  QLoRA r = 16: 4-bit block weights, the rest 16-bit {q4 / GB:.1f} GB + LoRA {16 * r16 / GB:.2f} GB "
      f"= {(q4 + 16 * r16) / GB:6.1f} GB")
dq = 0.127 / 8 * L * per_layer                                   # QLoRA's double-quantized block scales
print(f"  (plus NF4's block scales: 0.127 bits per weight with double quantization, {dq / GB:.2f} GB)")
print(f"\nOne r = 16 adapter stored in 16 bits: {2 * r16 / 1e6:.0f} MB, against {2 * total / GB:.1f} GB for a full copy")