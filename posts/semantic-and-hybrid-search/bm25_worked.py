from bm25 import BM25, tokens
from sections import sections

# BM25 worked for one query on this blog's 309 English sections: the
# query "paged KV cache", and the section that scores highest for it.
en = sections("en")
docs = [s["body"] for s in en]
bm = BM25(docs)
q = "paged KV cache"
best = int(bm.scores(q).argmax())
d = bm.docs[best]
print(f'query "{q}"; best section: "{en[best]["heading"]}" ({en[best]["post"]}), {bm.len[best]} words, '
      f"average {bm.avg:.0f}")
total = 0
for t in tokens(q):
    n = sum(1 for doc in bm.docs if t in doc)
    tf = d[t]
    norm = bm.k1 * (1 - bm.b + bm.b * bm.len[best] / bm.avg)
    part = bm.idf[t] * tf * (bm.k1 + 1) / (tf + norm)
    total += part
    print(f"  {t:6s} in {n:3d} of {len(docs)} sections: idf {bm.idf[t]:.2f}; appears {tf:2d} times here; "
          f"saturated tf {tf * (bm.k1 + 1) / (tf + norm):.2f}; contributes {part:.2f}")
print(f"  score {total:.2f}")
