from gpt import GPT, Config

# Where the parameters are, counted from the model and from the formula:
# per block 4 D^2 in attention (Q, K, V, output) and 8 D^2 in the SwiGLU MLP,
# plus the embedding table, which the unembedding shares.
c = Config(V=148)                                     # the character model trained below
model = GPT(c)
groups = {"embedding (shared with unembedding)": [model.embed],
          "attention, all blocks": [m for b in model.blocks for m in (b.qkv, b.proj)],
          "MLP, all blocks": [m for b in model.blocks for m in (b.w_gate, b.w_up, b.w_down)],
          "norm gains": [n for b in model.blocks for n in (b.n1, b.n2)] + [model.norm]}
total = sum(p.numel() for p in model.parameters())
for name, mods in groups.items():
    n = sum(p.numel() for m in mods for p in m.parameters())
    print(f"{name:37s} {n:>9,}  {n / total:5.1%}")
print(f"{'total':37s} {total:>9,}")
formula = lambda L, D, V: 12 * L * D ** 2 + V * D
print(f"formula 12 L D^2 + V D: {formula(c.L, c.D, c.V):,}  (L={c.L}, D={c.D}, V={c.V})")
print(f"with post 9's 2,048 BPE tokens instead: {formula(4, 192, 2048):,}, "
      f"embedding {2048 * 192 / formula(4, 192, 2048):.0%} of it")
print(f"GPT-2 small (L=12, D=768, V=50,257): {formula(12, 768, 50257) / 1e6:.1f}M by the formula; "
      f"124M reported, the rest mostly its 1,024 learned position vectors")