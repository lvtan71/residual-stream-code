"""Train, encode and decode on the 19-word corpus, printing every step."""

import regex as re

from bpe import (CORPUS, PAT, SPECIAL, count_pairs, decode, encode, encode_word, pretokenize,
                 show, to_bytes, train)


def row(tok):
    return "(" + ", ".join(show(t) for t in tok) + ")"


print("pre-tokens, in order:", " ".join(show(m.group().encode())
                                       for m in re.finditer(PAT, CORPUS)))
counts = pretokenize(CORPUS)
print(f"\n{len(counts)} distinct pre-tokens, {sum(counts.values())} occurrences")
for tok, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"  {row(tok):24s} {n}")

pairs = count_pairs(counts)
print(f"\n{len(pairs)} distinct adjacent pairs")
for p, n in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])):
    print(f"  {row(p):12s} {n}")

print("\n #   id  pairs at the top count         merge         new token  count")
step = []


def on_merge(pairs, best):
    top = pairs[best]
    tied = sorted(p for p, n in pairs.items() if n == top)
    k = len(step)
    step.append(best)
    print(f"{k + 1:2d}  {256 + k}  {' '.join(row(p) for p in tied):29s} "
          f"{show(best[0])} + {show(best[1]):9s} {show(best[0] + best[1]):10s} {top}")


vocab, merges = train(CORPUS, 256 + 8 + 1, specials=[SPECIAL], on_merge=on_merge)
print("\nvocab beyond the bytes:", {i: vocab[i] for i in range(256, len(vocab))})
print("merges:", merges)

rank = {p: i for i, p in enumerate(merges)}
to_id = {t: i for i, t in vocab.items()}
for word in (" newest", "lowest"):
    print(f"\nencode {show(word.encode())!r}")
    print(f"  start:  {row(to_bytes(word))}")
    k = 0

    def on_step(tok, cands, p):
        global k
        k += 1
        c = " ".join(f"{row(q)}#{rank[q]}" for q in cands)
        print(f"  iter {k}: {c:34s} -> apply rank {rank[p]} {row(p)}")

    out = encode_word(to_bytes(word), rank, on_step)
    print(f"  result: {row(out)}  ids {[to_id[t] for t in out]}")

for word in ("newer", "zzqq"):
    out = encode_word(to_bytes(word), rank)
    print(f"\n{word}: {row(out)}  ids {encode(word, vocab, merges)}")

text = "the lowest widest fox" + SPECIAL + "newest lower"
ids = encode(text, vocab, merges, specials=[SPECIAL])
print(f"\ntext:  {text!r}")
print("split:", [c for c in re.split("(" + re.escape(SPECIAL) + ")", text)])
print("ids:  ", ids)
print("tokens:", " | ".join(show(vocab[i]) for i in ids))
print(f"{len(text.encode())} bytes -> {len(ids)} tokens = "
      f"{len(text.encode()) / len(ids):.2f} bytes per token")
print("round trip:", decode(ids, vocab) == text)

emoji = "🙃"
ids = encode(emoji, vocab, merges)
print(f"\n{emoji!r} -> ids {ids} -> {decode(ids, vocab)!r}")
print(f"first two bytes only {ids[:2]} -> {decode(ids[:2], vocab)!r}")
