import torch

# The famous test: does b - a + c land nearest to d? ("king - man + woman ~ queen")
m = torch.load("word2vec.pt")
vocab, E = m["vocab"], m["E"]
index = {w: i for i, w in enumerate(vocab)}
tests = [("encoder", "decoder", "input", "output"), ("keys", "values", "key", "value"),
         ("prefill", "decode", "prompt", "generation"),
         ("layer", "layers", "token", "tokens"), ("head", "heads", "weight", "weights"),
         ("key", "keys", "value", "values")]
for a, b, c, d in tests:
    if not all(w in index for w in (a, b, c, d)):
        print(f"{a}:{b} :: {c}:{d}  (a word is missing from the vocabulary)")
        continue
    q = E[index[b]] - E[index[a]] + E[index[c]]
    sims = torch.nn.functional.normalize(q, dim=0) @ E.T
    for w in (a, b, c):
        sims[index[w]] = -1                          # the usual rule: exclude the question words
    ranked = sims.argsort(descending=True)
    rank = (ranked == index[d]).nonzero().item() + 1
    print(f"{a} : {b} :: {c} : ?   top answer '{vocab[ranked[0]]}', the expected '{d}' is ranked {rank} "
          f"of {len(vocab) - 3:,}")