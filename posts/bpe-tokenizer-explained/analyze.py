"""What the TinyStories tokenizers learned, and how well they compress.

Reads ts_train_* and ts_valid_*, written by tinystories.py.
"""

from collections import Counter

import tiktoken

from bpe import SPECIAL, decode, encode, show
from data import tinystories
from files import load

vocab, merges = load("ts_train")
small_vocab, small_merges = load("ts_valid")
gpt2 = tiktoken.get_encoding("gpt2")

# Held out: the training tokenizer never saw the validation file.
docs = [d.strip() for d in open(tinystories("valid"), encoding="utf-8").read().split(SPECIAL)]
docs = [d for d in docs if d][:200]
n_bytes = sum(len(d.encode()) for d in docs)
ids = [encode(d, vocab, merges) for d in docs]
ours, theirs = sum(map(len, ids)), sum(len(gpt2.encode(d)) for d in docs)
print(f"200 held-out stories, {n_bytes:,} bytes")
print(f"  this tokenizer (10,000):  {ours:,} tokens, {n_bytes / ours:.2f} bytes per token")
print(f"  GPT-2's (50,257):         {theirs:,} tokens, {n_bytes / theirs:.2f} bytes per token")
print(f"  round trip exact: {sum(decode(i, vocab) == d for i, d in zip(ids, docs))} of 200")

print("\nfirst 12 merges:", "  ".join(show(a + b) for a, b in merges[:12]))
learned = [vocab[i] for i in range(256, 256 + len(merges))]
print("ten longest tokens:", ", ".join(show(t) for t in sorted(learned, key=lambda t: (-len(t), t))[:10]))
spaced = sum(t[:1] == b" " for t in learned)
print(f"{spaced:,} of {len(learned):,} learned tokens start with a space "
      f"({spaced / len(learned):.0%})")
lengths = Counter(len(t) for t in learned)
print("learned tokens by length in bytes:",
      ", ".join(f"{n}: {lengths[n]}" for n in sorted(lengths)))

for text in ("Once upon a time, there was a little boat.",):
    print(f"\n{text!r}")
    print("  this tokenizer:", " | ".join(show(vocab[i]) for i in encode(text, vocab, merges)))
    print("  GPT-2:         ",
          " | ".join(show(gpt2.decode_single_token_bytes(i)) for i in gpt2.encode(text)))

shared = set(vocab.values()) & set(small_vocab.values())
first = next(i for i, (a, b) in enumerate(zip(merges, small_merges)) if a != b)
print(f"\ntrained on 1% of the data (the 22 MB validation file): {len(shared):,} of "
      f"{len(vocab):,} vocabulary entries shared")
print(f"merge lists first differ at merge {first + 1}: "
      f"{show(merges[first][0])} + {show(merges[first][1])} (full data) vs "
      f"{show(small_merges[first][0])} + {show(small_merges[first][1])} (1%)")
