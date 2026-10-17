import math
from collections import Counter

# The chain rule, with the simplest model that obeys it: each character's
# probability given only the one before it (a bigram model), counted on the
# blog's training text with add-one smoothing.
train, val = open("blog_train.txt").read(), open("blog_val.txt").read()
V = len(set(train)) + 1
first, pair = Counter(train), Counter(train[i:i + 2] for i in range(len(train) - 1))
p = lambda c, prev: (pair[prev + c] + 1) / (first[prev] + V)

text = "the cache"
logp = math.log(first[text[0]] / len(train))
print(f"p('{text[0]}') = {math.exp(logp):.4f}")
for prev, c in zip(text, text[1:]):
    logp += math.log(p(c, prev))
    print(f"p('{c}' | '{prev}') = {p(c, prev):.4f}")
print(f"p('{text}') = product = {math.exp(logp):.3e};  -log2 p = {-logp / math.log(2):.1f} bits "
      f"for {len(text)} characters")

# Perplexity of the bigram model on the held-out posts.
nll = -sum(math.log(p(c, prev)) for prev, c in zip(val, val[1:])) / (len(val) - 1)
print(f"bigram model on held-out posts: {nll:.3f} nats per character, perplexity {math.exp(nll):.1f}, "
      f"{nll / math.log(2):.3f} bits per character")