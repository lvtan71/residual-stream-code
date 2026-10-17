import collections
import math
import re
import unicodedata
import numpy as np
from sentence_transformers import SentenceTransformer
from sections import sections

# Keyword (BM25), dense and hybrid retrieval on this blog's own sections:
# each heading is a query whose one relevant document is its section's text.
# Headings that occur more than once in a language are left out (ambiguous).
def unique(secs):
    count = collections.Counter(s["heading"].lower() for s in secs)
    return [s for s in secs if count[s["heading"].lower()] == 1]
en, vi = sections("en"), sections("vi")
def strip_marks(t):                                  # Vietnamese typed without diacritics
    t = t.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


# Cross-lingual pairs: a Vietnamese heading and the English section of the
# same post whose heading it translates, matched as mutual best matches by
# the multilingual model's heading similarity (the twins' sections don't
# always line up one to one).
def cross_pairs():
    by_post = lambda secs: {p: [s for s in secs if s["post"] == p and s in unique(secs)] for p in {s["post"] for s in secs}}
    EN, VI = by_post(en), by_post(vi)
    pairs = []
    for p in VI:
        if not EN.get(p) or not VI[p]:
            continue
        S = models["MiniLM (multilingual)"].encode([v["heading"] for v in VI[p]], normalize_embeddings=True) @ \
            models["MiniLM (multilingual)"].encode([e["heading"] for e in EN[p]], normalize_embeddings=True).T
        for i in range(len(VI[p])):
            j = int(S[i].argmax())
            if int(S[:, j].argmax()) == i:
                pairs.append((VI[p][i]["heading"], en.index(EN[p][j])))
    return pairs

def tokens(t):
    return re.findall(r"\w+", t.lower())

class BM25:
    """Okapi BM25 (Robertson and Zaragoza, 2009), k1 = 1.5, b = 0.75."""
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [collections.Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(d.values()) for d in self.docs]); self.avg = self.len.mean()
        df = collections.Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
        self.k1, self.b = k1, b
    def scores(self, query):
        s = np.zeros(len(self.docs))
        for t in set(tokens(query)):
            if t not in self.idf:
                continue
            tf = np.array([d[t] for d in self.docs])
            s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len / self.avg))
        return s

models = {"MiniLM (English)": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", device="cpu"),
          "MiniLM (multilingual)": SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42", device="cpu")}
def dense(model, docs):
    """One vector per section. The models read at most 256 (English) and
    128 (multilingual) word pieces, so long sections are cut off."""
    D = models[model].encode(docs, batch_size=64, normalize_embeddings=True)
    return lambda q: D @ models[model].encode([q], normalize_embeddings=True)[0]

def dense_passages(model, docs, words=80):
    """One vector per 80-word passage; a section scores its best passage."""
    owner, passages = [], []
    for i, d in enumerate(docs):
        w = d.split()
        for j in range(0, len(w), words):
            owner.append(i); passages.append(" ".join(w[j:j + words]))
    P = models[model].encode(passages, batch_size=64, normalize_embeddings=True)
    owner = np.array(owner)
    def f(q):
        s = P @ models[model].encode([q], normalize_embeddings=True)[0]
        best = np.full(len(docs), -np.inf)
        np.maximum.at(best, owner, s)
        return best
    return f

def rrf(*score_fns, k=60, weights=None):
    """Reciprocal rank fusion (Cormack et al., 2009): sum of 1 / (k + rank),
    optionally weighted per ranker."""
    weights = weights or [1] * len(score_fns)
    def f(q):
        total = 0
        for fn, wt in zip(score_fns, weights):
            order = np.argsort(-fn(q)); rank = np.empty_like(order); rank[order] = np.arange(1, len(order) + 1)
            total = total + wt / (k + rank)
        return total
    return f

def evaluate(name, fn, queries):
    ranks = []
    for q, target in queries:
        s = fn(q)
        ranks.append(int((s >= s[target]).sum()))                     # its rank; ties counted against it
    r = np.array(ranks)
    print(f"  {name:38s} recall@1 {np.mean(r == 1):5.0%}  recall@10 {np.mean(r <= 10):5.0%}  "
          f"MRR {np.mean(1 / r):.3f}  nDCG@10 {np.mean(np.where(r <= 10, 1 / np.log2(r + 1), 0)):.3f}")
    return 1 / r

def paired(name, a, b):
    """MRR difference a - b, with a 95% bootstrap interval over queries."""
    rng = np.random.default_rng(0)
    d = a - b
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name}: MRR difference {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
