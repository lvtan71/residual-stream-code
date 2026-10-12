"""The same BPE as bpe.py, made fast without changing a single merge.

Three changes: read the corpus in parallel chunks cut only at a special token,
update only the pre-tokens that hold the merged pair, and (optionally) keep the
pair counts in a heap instead of scanning them all for the maximum.
"""

import heapq
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

from bpe import merge, pretokenize


def boundaries(path, n, special=b"<|endoftext|>"):
    """n roughly equal byte ranges, each moved forward to start at a special token."""
    size = os.path.getsize(path)
    cuts = [size * i // n for i in range(n)] + [size]
    with open(path, "rb") as f:
        for i in range(1, n):
            pos = cuts[i]
            while True:
                f.seek(pos)
                data = f.read(1 << 20)
                j = data.find(special)
                if j >= 0:
                    cuts[i] = pos + j
                    break
                if len(data) < 1 << 20:  # reached the end of the file
                    cuts[i] = size
                    break
                pos += len(data) - len(special)  # a token may straddle two reads
    return sorted(set(cuts))


def count_range(args):
    path, start, end, specials = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")  # strict: a bad cut would raise
    return pretokenize(text, specials)


def pretokenize_file(path, specials, workers):
    cuts = boundaries(path, workers)
    jobs = [(path, a, b, specials) for a, b in zip(cuts, cuts[1:])]
    if workers == 1:
        return count_range(jobs[0])
    with Pool(workers) as pool:
        total = Counter()
        for part in pool.imap_unordered(count_range, jobs):
            total.update(part)
    return total


class Desc:
    """Wraps a pair so that heapq's min-heap pops the greatest pair first."""
    __slots__ = ("p",)

    def __init__(self, p):
        self.p = p

    def __lt__(self, other):
        return self.p > other.p


def train_counts(counts, vocab_size, specials=(), use_heap=False, times=None):
    """Learn merges from a pre-token table, touching only rows that change."""
    words = [list(t) for t in counts]
    freqs = list(counts.values())
    pairs, where = Counter(), defaultdict(set)  # pair -> total, pair -> rows holding it
    for i, (w, f) in enumerate(zip(words, freqs)):
        for p in zip(w, w[1:]):
            pairs[p] += f
            where[p].add(i)
    heap = [(-n, Desc(p), p) for p, n in pairs.items()] if use_heap else None
    if use_heap:
        heapq.heapify(heap)
    vocab = {i: bytes([i]) for i in range(256)}
    merges, t_pick, t_apply = [], 0.0, 0.0
    while len(vocab) < vocab_size - len(specials) and pairs:
        t0 = time.perf_counter()
        if use_heap:
            while True:  # lazy deletion: skip entries whose count has changed
                n, _, best = heapq.heappop(heap)
                if pairs.get(best) == -n:
                    break
        else:
            best = max(pairs, key=lambda p: (pairs[p], p))
        t1 = time.perf_counter()
        new = best[0] + best[1]
        changed = set()
        for i in list(where[best]):  # copy: the loop edits these sets
            w, f = words[i], freqs[i]
            for p in zip(w, w[1:]):
                pairs[p] -= f
                changed.add(p)
            w = list(merge(w, best, new))
            for p in zip(w, w[1:]):
                pairs[p] += f
                where[p].add(i)
                changed.add(p)
            words[i] = w
        for p in changed:
            if pairs[p] <= 0:
                del pairs[p]
            elif use_heap:
                heapq.heappush(heap, (-pairs[p], Desc(p), p))
        del where[best]
        merges.append(best)
        vocab[len(vocab)] = new
        t_apply += time.perf_counter() - t1
        t_pick += t1 - t0
    for s in specials:
        vocab[len(vocab)] = s.encode("utf-8")
    if times is not None:
        times.update(pick=t_pick, apply=t_apply)
    return vocab, merges
