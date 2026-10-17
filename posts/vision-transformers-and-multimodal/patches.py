# An image as a sequence. ViT-B/16 on a 224x224 RGB image: cut it into
# 16x16 patches, flatten each (16 * 16 * 3 numbers), and project each to the
# model's width with one shared linear layer, the patch embedding.
H = W = 224; P = 16; C = 3; D = 768
n = (H // P) * (W // P)
print(f"{H}x{W} image, {P}x{P} patches: {H // P} x {W // P} = {n} patches, each {P * P * C} numbers")
print(f"patch embedding: a {P * P * C} x {D} matrix + bias = {P * P * C * D + D:,} parameters")
print(f"sequence for the Transformer: {n} patch tokens + 1 [CLS] = {n + 1} positions, attention over {(n + 1) ** 2:,} pairs")
for res, p in ((224, 14), (336, 14), (448, 14), (1024, 16)):
    print(f"  {res}x{res} with {p}x{p} patches: {(res // p) ** 2:,} tokens")

# In a vision-language model the patch embeddings, after a projector, become
# tokens in the language model's context, and cost KV cache like text.
# LLaVA-1.5: a CLIP ViT-L/14 at 336x336 gives 576 image tokens, fed to a 7B
# language model (32 layers, width 4096, full multi-head attention, 16-bit).
kv_per_token = 32 * 2 * 4096 * 2                 # layers x (K and V) x width x bytes
print(f"\nLLaVA-1.5 (7B language model): {kv_per_token / 2 ** 10:.0f} KiB of KV cache per token")
print(f"  one image, 576 tokens: {576 * kv_per_token / 2 ** 20:.0f} MiB; a 20-image conversation: "
      f"{20 * 576 * kv_per_token / 2 ** 30:.1f} GiB")
print(f"  as text, 576 tokens is about {int(576 * 0.75)} English words")
