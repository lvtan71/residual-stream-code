"""Train a 10,000-token BPE on TinyStories and time each stage.

python tinystories.py valid   # 22 MB, seconds
python tinystories.py train   # 2.2 GB, a few minutes; time it on a quiet machine

Trains twice from the same pre-token table, once scanning for the maximum pair
and once with a heap, and saves the tokenizer as ts_<split>_vocab.json and
ts_<split>_merges.txt.
"""

import os
import platform
import resource
import sys
import time

from bpe import SPECIAL
from data import tinystories
from fast_bpe import pretokenize_file, train_counts
from files import save

if __name__ == "__main__":
    split = sys.argv[1]
    path = tinystories(split)
    workers = min(8, os.cpu_count())
    print(f"{os.path.basename(path)}: {os.path.getsize(path):,} bytes; {workers} workers; "
          f"{platform.machine()}, {os.cpu_count()} cores, Python {platform.python_version()}")

    t0 = time.perf_counter()
    counts = pretokenize_file(path, [SPECIAL], workers)
    t_pre = time.perf_counter() - t0
    print(f"pre-tokenize: {t_pre:6.1f} s  ({len(counts):,} distinct pre-tokens, "
          f"{sum(counts.values()):,} in all)")

    for use_heap in (False, True):
        times = {}
        vocab, merges = train_counts(counts, 10_000, [SPECIAL], use_heap, times)
        total = t_pre + times["pick"] + times["apply"]
        print(f"{'heap' if use_heap else 'scan'}: pick the pair {times['pick']:6.1f} s, "
              f"apply the merge {times['apply']:5.1f} s; with pre-tokenizing {total:6.1f} s "
              f"(pre-tokenize {t_pre / total:.0%}, pick {times['pick'] / total:.0%}, "
              f"apply {times['apply'] / total:.0%})")
        if use_heap:
            assert merges == scan_merges
        scan_merges = merges

    unit = 2**20 if sys.platform == "darwin" else 2**10  # ru_maxrss: bytes on macOS, KB on Linux
    self_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / unit
    child_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / unit
    print(f"{len(merges):,} merges; vocabulary {len(vocab):,}; {SPECIAL} is ID "
          f"{len(vocab) - 1:,}")
    print(f"peak memory: {self_mb:,.0f} MB in the main process, {child_mb:,.0f} MB in the "
          f"largest worker")
    save(vocab, merges, f"ts_{split}")
