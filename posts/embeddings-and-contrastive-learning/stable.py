from collections import Counter
import torch

# Which neighbours are stable? Top 6 by cosine in each of three word2vec seeds.
seeds = [torch.load(f"word2vec{'' if s == 0 else s}.pt") for s in (0, 1, 2)]
for w in ("attention", "cache", "gradient", "token", "vietnamese", "gpu"):
    c = Counter()
    for m in seeds:
        vocab, E = m["vocab"], m["E"]
        i = vocab.index(w)
        c.update(vocab[j] for j in (E @ E[i]).argsort(descending=True)[1:7].tolist())
    print(f"{w:11s} in all 3 seeds: {', '.join(x for x, n in c.items() if n == 3) or '-'};  "
          f"in 2: {', '.join(x for x, n in c.items() if n == 2) or '-'}")