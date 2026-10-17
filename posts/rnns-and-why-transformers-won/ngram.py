import math
from collections import Counter

# Baselines with no neural network at all: predict the next character from
# the previous k characters by counting, with add-one smoothing.
train, val = open("blog_train.txt").read(), open("blog_val.txt").read()
V = len(set(train)) + 1                                   # + an "unseen" character
for k in (0, 1, 2, 3, 4):
    ctx, pair = Counter(), Counter()
    for i in range(k, len(train)):
        ctx[train[i - k:i]] += 1
        pair[train[i - k:i + 1]] += 1
    nll = -sum(math.log((pair[val[i - k:i + 1]] + 1) / (ctx[val[i - k:i]] + V))
               for i in range(k, len(val))) / (len(val) - k)
    print(f"previous {k} characters: validation {nll:.3f} nats/char = {nll / math.log(2):.3f} bits/char")