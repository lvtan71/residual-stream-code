"""Byte-level BPE from scratch: training, encoding and decoding.

This is the minimal version. It recounts every pair on every merge, which is
correct and slow. fast_bpe.py learns the same merges with an index of which
pre-tokens hold each pair and with parallel reading.
"""

from collections import Counter

import regex as re  # the stdlib re has no \p{L}

# GPT-2's pre-tokenizer with its contractions grouped (tiktoken pull request 234).
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPECIAL = "<|endoftext|>"

# The example corpus from Sennrich et al. (2016), as running text.
CORPUS = ("low low low low low\n"
          "lower lower widest widest widest\n"
          "newest newest newest newest newest newest\n")


def split_special(text, specials):
    """Split text around special tokens, keeping them as their own chunks."""
    if not specials:  # re.split("()", text) would cut between every character
        return [text]
    longest_first = sorted(specials, key=len, reverse=True)
    return re.split("(" + "|".join(re.escape(s) for s in longest_first) + ")", text)


def to_bytes(s):
    """A string as a tuple of one-byte tokens."""
    return tuple(bytes([b]) for b in s.encode("utf-8"))


def pretokenize(text, specials=()):
    """The corpus as a table: {pre-token as a tuple of bytes: how often it occurs}."""
    counts = Counter()
    for chunk in split_special(text, specials):
        if chunk in specials:  # special tokens take no part in training
            continue
        for m in re.finditer(PAT, chunk):
            counts[to_bytes(m.group())] += 1
    return counts


def count_pairs(counts):
    pairs = Counter()
    for tok, freq in counts.items():
        for pair in zip(tok, tok[1:]):  # adjacent pairs, duplicates kept
            pairs[pair] += freq
    return pairs


def merge(tok, pair, new):
    """Replace every occurrence of pair in tok, left to right, without overlap."""
    out, i = [], 0
    while i < len(tok):
        if i < len(tok) - 1 and (tok[i], tok[i + 1]) == pair:
            out.append(new)
            i += 2
        else:
            out.append(tok[i])
            i += 1
    return tuple(out)


def train(text, vocab_size, specials=(), on_merge=None):
    counts = pretokenize(text, specials)
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    while len(vocab) < vocab_size - len(specials):
        pairs = count_pairs(counts)  # naive: recount everything every round
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))  # count, then the greater pair
        if on_merge:
            on_merge(pairs, best)
        new = best[0] + best[1]
        merged = Counter()
        for tok, freq in counts.items():
            merged[merge(tok, best, new)] += freq  # two rows can become one
        counts = merged
        merges.append(best)
        vocab[len(vocab)] = new
    for s in specials:  # appended after the last merge
        vocab[len(vocab)] = s.encode("utf-8")
    return vocab, merges


def encode_word(tok, rank, on_step=None):
    """Apply learned merges to one pre-token, earliest-learned first."""
    while True:
        cands = [p for p in zip(tok, tok[1:]) if p in rank]
        if not cands:
            return tok
        p = min(cands, key=rank.get)  # lowest rank wins; position does not matter
        if on_step:
            on_step(tok, cands, p)
        tok = merge(tok, p, p[0] + p[1])


def encode(text, vocab, merges, specials=()):
    rank = {p: i for i, p in enumerate(merges)}
    to_id = {tok: i for i, tok in vocab.items()}
    ids = []
    for chunk in split_special(text, specials):
        if chunk in specials:  # straight to its ID, never merged
            ids.append(to_id[chunk.encode("utf-8")])
            continue
        for m in re.finditer(PAT, chunk):
            ids += [to_id[t] for t in encode_word(to_bytes(m.group()), rank)]
    return ids


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def show(tok):
    """Bytes as readable text: space as ␣, newline as ⏎, other bytes as hex."""
    out = []
    for b in tok:
        if b == 32:
            out.append("␣")
        elif b == 10:
            out.append("⏎")
        elif 33 <= b < 127:
            out.append(chr(b))
        else:
            out.append(f"<{b:02X}>")
    return "".join(out)
