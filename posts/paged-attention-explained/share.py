import torch
from transformers import DynamicCache
from lm import model, tok, DEVICE
from paged import Pool, Sequence, step, BLOCK

# Parallel sampling: four answers to one prompt. With paging the prompt is
# computed and stored once; the four sequences share its blocks, and a
# shared block is copied only when a sequence first writes into it. Each
# sample uses its own seeded generator, so the same four answers can be
# drawn again without sharing, from four separate contiguous caches.
cfg = model.config
pool = Pool(256, cfg.num_hidden_layers, cfg.num_key_value_heads,
            cfg.hidden_size // cfg.num_attention_heads, model.dtype, DEVICE)
messages = [{"role": "system", "content": "You are the Notebook Assistant. Answer questions about the "
             "Residual Stream blog in two or three sentences."},
            {"role": "user", "content": "Why does a KV cache need its own memory manager?"}]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                 return_dict=True)["input_ids"][0].to(DEVICE)
N, NEW, TEMP = 4, 64, 0.8

def sample(logits, gen):
    probs = torch.softmax(logits.float().cpu() / TEMP, -1)
    return int(torch.multinomial(probs, 1, generator=gen))

first = Sequence(pool)
logits = step(model, first, prompt)
print(f"prompt: {len(prompt)} tokens in {len(first.table)} blocks, the last one "
      f"{len(prompt) % BLOCK} of {BLOCK} full")
seqs = [first] + [first.fork() for _ in range(N - 1)]
gens = [torch.Generator().manual_seed(i) for i in range(N)]
paged = [[sample(logits, g)] for g in gens]
for t in range(NEW - 1):
    for s, out, g in zip(seqs, paged, gens):
        out.append(sample(step(model, s, torch.tensor(out[-1:], device=DEVICE)), g))
used = len(set(b for s in seqs for b in s.table))
alone = sum(len(s.table) for s in seqs)
print(f"{N} samples of {NEW} tokens: {used} blocks in use, against {alone} without sharing "
      f"({1 - used / alone:.0%} fewer); {pool.copies} copy-on-write copies")
print("tables:", *[s.table for s in seqs], sep="\n  ")

same = []
with torch.no_grad():
    for i in range(N):
        g, cache, ids, out = torch.Generator().manual_seed(i), DynamicCache(), prompt, []
        for t in range(NEW):
            out.append(sample(model(ids[None], past_key_values=cache, use_cache=True).logits[0, -1], g))
            ids = torch.tensor(out[-1:], device=DEVICE)
        same.append(out == paged[i])
print(f"each sample the same as without sharing: {same}")
for out in paged:
    print(" ", repr(tok.decode(out)[:90]))
