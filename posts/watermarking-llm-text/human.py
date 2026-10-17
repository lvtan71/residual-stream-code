import glob
import re
import math
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

# False positives on human text: every 200-token window of this blog's
# posts (prose only, code and math removed), scored with the same key as
# wm.py. Only the tokenizer is needed; the detector never runs the model.
src = open("wm.py").read()
exec(src[src.index("PATH = "):src.index("tok = ")])                  # PATH, KEY, GAMMA, N from wm.py
tok, V = AutoTokenizer.from_pretrained(PATH), AutoConfig.from_pretrained(PATH).vocab_size
exec(src[src.index("_green = {}"):src.index("@torch.no_grad()")])   # green() and z_score() from wm.py
posts = sorted(glob.glob("content/posts/*.mdx"))
def z_unique(ids):
    """The same test, scoring each (previous token, token) pair only once:
    a repeated pair repeats the same green outcome, so it isn't new evidence."""
    pairs = list(dict.fromkeys(zip(ids[:-1], ids[1:])))
    hits = sum(bool(green(p)[t]) for p, t in pairs)
    return (hits - GAMMA * len(pairs)) / math.sqrt(len(pairs) * GAMMA * (1 - GAMMA))

zs, zu = [], []
for path in posts:
    body = re.sub(r"```.*?```|\$\$.*?\$\$|^import .*?$|<[^>]+>", " ", open(path).read().split("---", 2)[-1], flags=re.S | re.M)
    ids = tok(" ".join(body.split())).input_ids
    for start in range(0, len(ids) - N - 1, N):
        zs.append(z_score(ids[start:start + N + 1])[0]); zu.append(z_unique(ids[start:start + N + 1]))
print(f"{len(zs)} windows of {N} tokens from {len(posts)} posts; a standard normal has mean 0, std 1")
for name, z in (("every pair", np.array(zs)), ("unique pairs only", np.array(zu))):
    print(f"{name:18s} z mean {z.mean():5.2f} (standard error {z.std() / np.sqrt(len(z)):.3f}), std {z.std():.2f}, "
          f"max {z.max():.2f}; above 4: {(z > 4).sum()}; above 2: {(z > 2).mean():.1%} (normal: 2.3%)")