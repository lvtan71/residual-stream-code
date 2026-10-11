import torch
from transformers import DynamicCache
from lm import model, tok, encode, DEVICE
from paged import Pool, Sequence, step, BLOCK

# Does paging change what the model computes? Two prompts decoded greedily
# for 100 tokens each, taking turns, so their blocks interleave in one
# pool; then the same two decoded with Hugging Face's contiguous cache.
cfg = model.config
pool = Pool(64, cfg.num_hidden_layers, cfg.num_key_value_heads,
            cfg.hidden_size // cfg.num_attention_heads, model.dtype, DEVICE)
per_block = BLOCK * cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * pool.k.shape[-1] * pool.k.element_size()
print(f"one block: {BLOCK} tokens x {cfg.num_hidden_layers} layers x K and V x {cfg.num_key_value_heads} heads "
      f"x {pool.k.shape[-1]} dimensions x {pool.k.element_size()} bytes = {per_block:,} bytes")
prompts = ["The KV cache stores", "Paged attention borrows an idea from operating systems:"]
seqs = [Sequence(pool) for _ in prompts]
paged = [[] for _ in prompts]
nxt = [encode(p) for p in prompts]
for t in range(100):
    for i, s in enumerate(seqs):                         # take turns, one step each
        logits = step(model, s, nxt[i])
        paged[i].append(int(logits.argmax()))
        nxt[i] = torch.tensor([paged[i][-1]], device=DEVICE)

with torch.no_grad():
    for i, p in enumerate(prompts):
        ids, cache, out = encode(p), DynamicCache(), []
        for t in range(100):
            logits = model(ids[None], past_key_values=cache, use_cache=True).logits[0, -1]
            out.append(int(logits.argmax()))
            ids = torch.tensor([out[-1]], device=DEVICE)
        print(f"prompt {i}: {seqs[i].n} tokens in blocks {seqs[i].table}; "
              f"same 100 tokens as the contiguous cache: {out == paged[i]}")
print(f"block size {BLOCK}; {len(pool.free)} of 64 blocks free")
print(repr(tok.decode(paged[1][:40])))
