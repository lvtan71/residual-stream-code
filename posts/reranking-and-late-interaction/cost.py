import numpy as np
from transformers import AutoTokenizer
from sections import sections

# Cost per query of the three designs, for this blog's 309 English
# sections, with a MiniLM-sized encoder: 22.7M parameters, about 11M of
# them the token-embedding table (a lookup), so roughly 2 x 11M FLOPs per
# token for the rest, ignoring attention. Queries are the section headings.
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
secs = sections("en")
full = np.array([len(tok(s["body"]).input_ids) for s in secs])
read = np.minimum(full, 512)                                   # what a 512-token encoder reads
q = np.mean([len(tok(s["heading"]).input_ids) for s in secs])
N, flop_tok, K = len(secs), 2 * 11e6, 50
print(f"{N} sections, {full.mean():.0f} tokens each on average ({read.mean():.0f} read at most 512); "
      f"queries average {q:.1f} tokens")
bi = q * flop_tok + N * 384 * 2
cross = K * (q + read.mean()) * flop_tok
late = q * flop_tok + N * q * full.mean() * 128 * 2
print(f"bi-encoder: encode the query, then {N} dot products           {bi / 1e6:10.1f} MFLOPs per query")
print(f"cross-encoder, top {K}: {K} passes over query + section        {cross / 1e6:10.1f} MFLOPs per query")
print(f"late interaction (ColBERT, 128-d): every query x section token {late / 1e6:10.1f} MFLOPs per query")
print(f"  roughly K (q + L) / q = {K * (q + read.mean()) / q:,.0f}: the cross-encoder's multiple of one query encoding")
for n_docs in (1e4, 1e6):
    print(f"  at {n_docs:,.0f} documents: bi-encoder {(q * flop_tok + n_docs * 384 * 2) / 1e9:6.2f} GFLOPs, "
          f"MaxSim over every document {(q * flop_tok + n_docs * q * full.mean() * 256) / 1e9:8.1f} GFLOPs")
print("\nindex size, every token indexed, 16-bit numbers:")
print(f"  bi-encoder, one 384-d vector per section:              {N * 384 * 2 / 2**20:7.2f} MiB")
print(f"  ColBERT, one 128-d vector per token:                   {full.sum() * 128 * 2 / 2**20:7.2f} MiB")
for b in (2, 1):
    print(f"  ColBERTv2, 4-byte centroid id + {b}-bit residuals:       {full.sum() * (4 + 128 * b / 8) / 2**20:7.2f} MiB")