"""The same sentences in English and Vietnamese: how many tokens does each cost?

2,000 sentence pairs from the OPUS-100 en-vi test split (mostly film
subtitles), counted with OpenAI's tokenizers and with this post's
TinyStories tokenizer (ts_train_*, written by tinystories.py).
"""

import pandas as pd
import tiktoken

from bpe import encode
from data import opus_en_vi
from files import load

pairs = pd.read_parquet(opus_en_vi())["translation"].tolist()
en = "\n".join(p["en"] for p in pairs)
vi = "\n".join(p["vi"] for p in pairs)
print(f"{len(pairs):,} sentence pairs")
print(f"{'':10s} {'characters':>11s} {'UTF-8 bytes':>12s} {'words':>8s}")
for name, t in (("English", en), ("Vietnamese", vi)):
    print(f"{name:10s} {len(t):11,d} {len(t.encode()):12,d} {len(t.split()):8,d}")

vocab, merges = load("ts_train")
counters = [(name, lambda t, e=tiktoken.get_encoding(name): len(e.encode(t)))
            for name in ("gpt2", "cl100k_base", "o200k_base")]
counters.append(("TinyStories", lambda t: len(encode(t, vocab, merges))))
print(f"\n{'tokenizer':12s} {'English':>8s} {'bytes/tok':>9s} {'Vietnamese':>11s} "
      f"{'bytes/tok':>9s} {'Vietnamese / English':>21s}")
for name, count in counters:
    ne, nv = count(en), count(vi)
    print(f"{name:12s} {ne:8,d} {len(en.encode()) / ne:9.2f} {nv:11,d} "
          f"{len(vi.encode()) / nv:9.2f} {nv / ne:20.2f}x")
