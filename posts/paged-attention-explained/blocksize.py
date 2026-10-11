import json
import math

# The block size, on the same requests as before. Smaller blocks waste
# less of the last block but give longer block tables to walk; and a cached
# prefix is reused only in whole blocks, so request B above, which shares
# its first 85 tokens with A, reuses fewer of them as blocks grow.
traces = {name: json.load(open(f"{name}_lengths.json")) for name in ("sharegpt", "alpaca")}
print("block   real tokens / reserved     blocks per request     B's 85 shared tokens")
print("size    ShareGPT     Alpaca        ShareGPT   Alpaca      reused from the cache")
for size in (1, 4, 8, 16, 32, 64, 128, 256):
    use, blocks = [], []
    for reqs in traces.values():
        held = reserved = 0
        for p, a in reqs:
            for t in range(1, a + 1):
                held += p + t
                reserved += size * math.ceil((p + t) / size)
        use.append(held / reserved)
        blocks.append(sum(math.ceil((p + a) / size) for p, a in reqs) / len(reqs))
    print(f"{size:4d}    {use[0]:7.1%}    {use[1]:7.1%}      {blocks[0]:8.1f} {blocks[1]:8.1f}      "
          f"{85 // size * size:3d}")
