# Post 7's Kneser-Ney (rnns-and-why-transformers-won), its first 40 lines: the
# definitions without the sweep, as the post's run imported it.
import math
from collections import Counter, defaultdict

# Interpolated Kneser-Ney smoothing (Chen and Goodman, 1998), with a fixed
# discount of 0.75. Instead of spreading unseen probability evenly, it backs
# off to a shorter context, and the shortest level counts how many different
# contexts a character follows, not how often it occurs.
train, val = open("blog_train.txt").read(), open("blog_val.txt").read()
chars = sorted(set(train))
V, D = len(chars) + 1, 0.75                                # + one "unseen" character

def build(order):
    count = [Counter() for _ in range(order + 1)]          # count[n][context + char], n = context length
    for n in range(order + 1):
        for i in range(n, len(train)):
            count[n][train[i - n:i + 1]] += 1
    # For lower orders, KN uses continuation counts: in how many distinct contexts does it appear?
    cont = [Counter() for _ in range(order + 1)]
    for n in range(order):
        for gram in count[n + 1]:
            cont[n][gram[1:]] += 1
    tables = []
    for n in range(order + 1):
        c = count[n] if n == order else cont[n]
        total, types = defaultdict(int), defaultdict(int)
        for gram, k in c.items():
            total[gram[:-1]] += k
            types[gram[:-1]] += 1
        tables.append((c, total, types))
    return tables

def prob(tables, ctx, ch):
    p = 1 / V                                              # the uniform floor
    for n in range(len(tables)):
        c, total, types = tables[n]
        h = ctx[len(ctx) - n:] if n else ""
        if total[h] == 0:
            continue                                        # context never seen: keep the shorter estimate
        p = max(c[h + ch] - D, 0) / total[h] + D * types[h] / total[h] * p
    return p
