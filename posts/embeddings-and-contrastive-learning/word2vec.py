import re
import sys
from collections import Counter
import torch
from split import train_text

# Skip-gram with negative sampling (Mikolov et al., 2013) on this blog's
# words: learn a vector per word such that words appearing near each other
# have a large dot product, and random pairs a small one. Trained only on
# text outside the held-out sections. Two simplifications from the original:
# frequent words aren't subsampled, and every distance in the window counts
# equally.
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
text = train_text.lower()
words = re.findall(r"[a-z][a-z0-9\-]+", text)
counts = Counter(words)
vocab = [w for w, c in counts.items() if c >= 5]
index = {w: i for i, w in enumerate(vocab)}
ids = torch.tensor([index[w] for w in words if w in index])
print(f"{len(words):,} words, {len(vocab):,} kept (5+ occurrences)")

torch.manual_seed(seed)
D, window, k = 64, 4, 5
center_emb = torch.nn.Embedding(len(vocab), D)
context_emb = torch.nn.Embedding(len(vocab), D)
torch.nn.init.normal_(center_emb.weight, 0, 0.1); torch.nn.init.zeros_(context_emb.weight)
freq = torch.tensor([counts[w] for w in vocab], dtype=torch.float) ** 0.75   # word2vec's sampling rule
opt = torch.optim.Adam(list(center_emb.parameters()) + list(context_emb.parameters()), lr=3e-3)
for step in range(4000):
    pos = torch.randint(window, len(ids) - window, (1024,))
    offset = torch.randint(1, window + 1, (1024,)) * (torch.randint(0, 2, (1024,)) * 2 - 1)
    c, o = ids[pos], ids[pos + offset]                                # a real (center, context) pair
    neg = torch.multinomial(freq, 1024 * k, replacement=True).view(1024, k)  # k random "contexts"
    vc = center_emb(c)
    pos_score = (vc * context_emb(o)).sum(-1)
    neg_score = torch.einsum("bd,bkd->bk", vc, context_emb(neg))
    # Logistic loss ("Learning Is Optimization"): real pairs toward 1, random pairs toward 0.
    loss = -(torch.nn.functional.logsigmoid(pos_score).mean()
             + torch.nn.functional.logsigmoid(-neg_score).sum(-1).mean())
    opt.zero_grad(); loss.backward(); opt.step()
print(f"final loss {loss.item():.3f} (at the start: (1 + k) log 2 = {(1 + k) * 0.6931:.2f})")

raw = center_emb.weight.detach()
E = torch.nn.functional.normalize(raw, dim=1)                          # unit length: dot = cosine
torch.save({"vocab": vocab, "E": E, "raw": raw, "counts": [counts[w] for w in vocab]},
           f"word2vec{'' if seed == 0 else seed}.pt")
for w in ("attention", "cache", "gradient", "token", "vietnamese", "gpu"):
    sims = E @ E[index[w]]
    best = [vocab[j] for j in sims.argsort(descending=True)[1:7]]
    print(f"{w:11s} -> {', '.join(best)}")
for a, b in (("cache", "caching"), ("token", "tokens"), ("cache", "cached")):
    rank = (E @ E[index[a]]).argsort(descending=True).tolist().index(index[b])
    print(f"cos({a}, {b}) = {(E[index[a]] @ E[index[b]]).item():.2f}, rank {rank} of {len(vocab):,}")