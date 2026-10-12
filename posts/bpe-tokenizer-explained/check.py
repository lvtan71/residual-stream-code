"""Check that the fast trainer gives exactly the naive trainer's merges."""

from bpe import SPECIAL, pretokenize, train
from data import tinystories
from fast_bpe import pretokenize_file, train_counts

if __name__ == "__main__":  # macOS starts worker processes by re-importing this file
    path = tinystories("valid")
    text = open(path, encoding="utf-8").read()
    sample = text[:500_000]
    _, naive = train(sample, 256 + 500 + 1, specials=[SPECIAL])
    _, fast = train_counts(pretokenize(sample, [SPECIAL]), 256 + 500 + 1, [SPECIAL])
    _, heap = train_counts(pretokenize(sample, [SPECIAL]), 256 + 500 + 1, [SPECIAL],
                           use_heap=True)
    print(f"first 500,000 characters, 500 merges: indexed == naive: {fast == naive}, "
          f"heap == naive: {heap == naive}")

    tables = {w: pretokenize_file(path, [SPECIAL], w) for w in (1, 2, 4, 8)}
    print(f"whole valid file, pre-token tables at 1, 2, 4, 8 workers identical: "
          f"{all(t == tables[1] for t in tables.values())} "
          f"({len(tables[1]):,} distinct pre-tokens)")
    _, scan = train_counts(tables[1], 10_000, [SPECIAL])
    _, heap = train_counts(tables[1], 10_000, [SPECIAL], use_heap=True)
    print(f"whole valid file, 10,000-token vocabulary: heap == scan: {heap == scan}")
