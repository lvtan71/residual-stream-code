import torch
from transformers import AutoModelForCausalLM
from lm import PATH, tok, DEVICE

# Where does attention go? For 1,024 tokens of this blog, the share of each
# layer's attention (averaged over heads and over every query position past
# the first 64) that lands on the very first token, on the 4 most recent
# tokens, and on everything in between.
# 32-bit floats: with the explicit ("eager") attention needed to read the
# weights out, 16-bit scores overflow in this model.
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32, attn_implementation="eager").to(DEVICE).eval()
text = open("blog_train.txt").read()
ids = torch.tensor([tok(text[200_000:206_000]).input_ids[:1024]], device=DEVICE)
with torch.no_grad():
    attn = model(ids, output_attentions=True).attentions          # per layer: (1, heads, T, T)
T = ids.shape[1]
rows = []
for layer, a in enumerate(attn):
    a = a[0, :, 64:, :].float().mean(0)                         # heads averaged; queries 64..T-1
    q = torch.arange(64, T, device=a.device)
    first = a[:, 0].mean().item()
    recent = torch.stack([a[i, max(0, int(qi) - 3):int(qi) + 1].sum() for i, qi in enumerate(q)]).mean().item()
    rows.append((layer, first, recent, 1 - first - recent))
for layer, first, recent, rest in rows:
    if layer % 3 == 0 or layer == len(rows) - 1:
        print(f"layer {layer:2d}: first token {first:5.1%}, last 4 tokens {recent:5.1%}, the other ~1,000 {rest:5.1%}")
print(f"averaged over all {len(rows)} layers: first token {sum(r[1] for r in rows) / len(rows):.1%}, "
      f"last 4 tokens {sum(r[2] for r in rows) / len(rows):.1%}; a uniform split would give the first token "
      f"{sum(1 / (q + 1) for q in range(64, T)) / (T - 64):.2%}")

# Per head rather than per layer (layers 1 to 26, 12 heads each).
per_head = torch.stack([a[0, :, 64:, 0].float().mean(-1) for a in attn[1:27]]).flatten()
print(f"heads in layers 1-26 putting over half their attention on the first token: "
      f"{int((per_head > 0.5).sum())} of {per_head.numel()}; under 5%: {int((per_head < 0.05).sum())}")

# Is it that token, or the first position? Four passages starting with
# different tokens, and the first one again with its first token moved to
# the end: which position gets the most attention, and how much.
def most_attended(ids):
    with torch.no_grad():
        att = model(torch.tensor([ids], device=DEVICE), output_attentions=True).attentions
    share = sum(a[0, :, 64:, :].float().mean((0, 1)) for a in att) / len(att)   # each key's average weight
    v, i = share.max(0)
    return int(i), tok.decode([ids[int(i)]]), v.item()

first = ids[0].tolist()
for name, x in [(f"passage at character {k:,}", tok(text[k:k + 6000]).input_ids[:1024]) for k in (100_000, 200_000, 300_000, 400_000)] \
               + [("first passage, first token moved to the end", first[1:] + first[:1])]:
    pos, token, share = most_attended(x)
    print(f"{name:44s} starts with {tok.decode(x[:1])!r:12s}: most attention on position {pos} ({token!r}), {share:.1%}")

# The digit: the first passage with one token put in front of it.
for lead in ("7", "3", "A", ".", "\n", "the"):
    x = tok(lead).input_ids + first
    pos, token, share = most_attended(x[:1024])
    print(f"first passage with {lead!r:6s} in front: most attention on position {pos} ({token!r}), {share:.1%}")
