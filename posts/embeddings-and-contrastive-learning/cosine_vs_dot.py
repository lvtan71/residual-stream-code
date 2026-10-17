import torch

# Dot product vs cosine on the raw word2vec vectors. The dot product is the
# cosine times both lengths, so a long vector rises in any list it's already
# near.
m = torch.load("word2vec.pt")
vocab, raw, counts = m["vocab"], m["raw"], torch.tensor(m["counts"], dtype=torch.float)
index = {w: i for i, w in enumerate(vocab)}
norms = raw.norm(dim=1)
r = torch.corrcoef(torch.stack([norms, counts.log()]))[0, 1]
print(f"vector length vs log word frequency: correlation {r:.2f}")
q = raw[index["cache"]]
for name, scores in [("dot product", raw @ q), ("cosine", (raw @ q) / (norms * q.norm()))]:
    scores[index["cache"]] = -1e9
    top = scores.argsort(descending=True)[:6]
    print(f"nearest to 'cache' by {name:11s}: " + ", ".join(f"{vocab[j]} ({norms[j]:.1f})" for j in top))
print(f"(vector lengths in brackets; the median word's is {norms.median():.1f})")