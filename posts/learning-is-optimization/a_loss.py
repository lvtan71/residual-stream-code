import numpy as np

# Words and o200k_base tokens for each English post on this blog, 2026-09-29
# (full MDX body, including math and code).
posts = {
    "attention-alternatives": (5880, 9793),
    "bpe-tokenizer-explained": (6638, 11489),
    "context-engineering-compaction-clearing-memory": (4336, 7104),
    "disciplined-ml-chess-outcomes": (2634, 4486),
    "kv-cache-explained": (3227, 5737),
    "paged-attention-explained": (2716, 4464),
    "residual-stream-explained": (2815, 4942),
    "sglang-radixattention-explained": (3143, 5140),
    "tensorrt-llm-explained": (3435, 6199),
    "transformer-attention-patterns": (3507, 6808),
    "vllm-architecture-explained": (2584, 4405),
    "vllm-vs-sglang-vs-tensorrt-llm": (1732, 3209),
}
words = np.array([w for w, _ in posts.values()]) / 1000   # thousands
tokens = np.array([t for _, t in posts.values()]) / 1000

# The simplest possible model: every post costs c tokens per word.
ratio = tokens / words

if __name__ == "__main__":
    mse = lambda c: ((ratio - c) ** 2).mean()
    mae = lambda c: np.abs(ratio - c).mean()
    grid = np.linspace(1.5, 2.0, 50001)
    print(f"L2 minimizer = {grid[np.argmin([mse(c) for c in grid])]:.4f}"
          f"   mean   = {ratio.mean():.4f}")
    lo, hi = np.sort(ratio)[5], np.sort(ratio)[6]   # the two middle values
    print(f"L1 is flat on [{lo:.4f}, {hi:.4f}]:"
          f" MAE = {mae(lo):.5f}, {mae((lo + hi) / 2):.5f}, {mae(hi):.5f}")
    print(f"median = {np.median(ratio):.4f}")