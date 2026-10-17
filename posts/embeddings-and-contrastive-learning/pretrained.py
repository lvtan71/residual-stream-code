import glob
import os
import re
import time
import torch
from sentence_transformers import SentenceTransformer
from sections import test_secs, W2V, to_ids, VIEW

# The same held-out task as contrastive.py, with two pretrained sentence
# encoders: all-MiniLM-L6-v2, trained contrastively on about a billion
# English sentence pairs, and a multilingual model distilled from an English
# one on translated pairs (Reimers and Gurevych, 2020).
models = {"all-MiniLM-L6-v2 (English)": "sentence-transformers/all-MiniLM-L6-v2",
          "multilingual MiniLM-L12": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}
REVISION = {"sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"}
first = [" ".join(s[:VIEW]) for s in test_secs]           # raw text: the same words the others see
last = [" ".join(s[-VIEW:]) for s in test_secs]

# Sections of the posts that have a Vietnamese twin, paired by position.
POSTS = "content"
def split(path):
    text = re.sub(r"\A---.*?---\n", "", open(path).read(), flags=re.S)
    skip = ("Table of contents", "Mục lục", "References", "Tài liệu tham khảo")   # headings only, or citation lists
    return [s.strip()[:1500] for s in re.split(r"\n## ", text)[1:] if not s.startswith(skip)]
pairs = []
for vi_path in sorted(glob.glob(f"{POSTS}/posts-vi/*.mdx")):
    en, vi = split(f"{POSTS}/posts/{os.path.basename(vi_path)}"), split(vi_path)
    if len(en) == len(vi):                                                # only posts whose headings line up
        pairs += list(zip(en, vi))
print(f"{len(test_secs)} held-out English sections; {len(pairs)} English/Vietnamese section pairs")

def recall(model, queries, docs):
    q = model.encode(queries, normalize_embeddings=True, convert_to_tensor=True)
    d = model.encode(docs, normalize_embeddings=True, convert_to_tensor=True)
    return ((q @ d.T).argmax(1) == torch.arange(len(q), device=q.device)).float().mean().item()

def spread(E):
    """Average cosine between *different* held-out English windows."""
    S = E @ E.T
    return S[~torch.eye(len(E), dtype=torch.bool)].mean().item()

w2v = torch.nn.functional.normalize(torch.stack([W2V[to_ids(s[:VIEW])].mean(0) for s in test_secs]), dim=-1)
print(f"averaged word2vec: average cosine between different held-out windows {spread(w2v):.2f}")
for name, path in models.items():
    model = SentenceTransformer(path, device="cpu", revision=REVISION[path])
    params = sum(p.numel() for p in model.parameters())
    same = recall(model, first, last)
    t0 = time.perf_counter()
    cross = recall(model, [e for e, _ in pairs], [v for _, v in pairs])
    secs = time.perf_counter() - t0
    E = model.encode(first, normalize_embeddings=True, convert_to_tensor=True)
    print(f"{name}: {params / 1e6:.0f}M parameters, reads up to {model.max_seq_length} tokens")
    print(f"    same-section recall@1 {same:.3f}   English -> Vietnamese recall@1 {cross:.3f} "
          f"({2 * len(pairs)} sections in {secs:.1f} s)   average cosine between different windows {spread(E):.2f}")