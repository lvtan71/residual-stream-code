"""What BPE learns with no pre-tokenizer: merges across word boundaries.

Treats each TinyStories story as one long sequence of bytes and runs the same
naive merge loop as bpe.py, against the same text pre-tokenized.
"""

from collections import Counter

from bpe import SPECIAL, count_pairs, merge, pretokenize, show, to_bytes
from data import tinystories

text = open(tinystories("valid"), encoding="utf-8").read()[:1_000_000]


def first_merges(counts, n):
    out = []
    for _ in range(n):
        pairs = count_pairs(counts)
        best = max(pairs, key=lambda p: (pairs[p], p))
        new = best[0] + best[1]
        counts = Counter({merge(t, best, new): f for t, f in counts.items()})
        out.append(new)
    return out


stories = Counter(to_bytes(s) for s in text.split(SPECIAL) if s)
print("no pre-tokenizer: ", "  ".join(show(t) for t in first_merges(stories, 20)))
print("GPT-2's regex:    ",
      "  ".join(show(t) for t in first_merges(pretokenize(text, [SPECIAL]), 20)))
