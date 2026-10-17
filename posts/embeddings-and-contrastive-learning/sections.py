import re
import torch
from split import train_secs, test_secs, VIEW

# The held-out split from split.py, plus word2vec vectors (trained without
# the held-out sections) and helpers shared by contrastive.py and pretrained.py.
def load(seed=0):
    m = torch.load(f"word2vec{'' if seed == 0 else seed}.pt")
    return m["vocab"], m["E"]

vocab, W2V = load(0)
index = {w: i for i, w in enumerate(vocab)}

def to_ids(toks, index=index):
    """The words word2vec knows, in order (vocab[0] if there are none)."""
    ids = [index[w] for t in toks for w in re.findall(r"[a-z][a-z0-9\-]+", t.lower()) if w in index]
    return torch.tensor(ids or [0])

def crop(sec, g):
    s = torch.randint(0, len(sec) - VIEW + 1, (1,), generator=g).item()
    return sec[s:s + VIEW]
