import collections
import math
import re
import numpy as np

# Okapi BM25, the keyword ranker of retrieval.py, and the tokenizer it uses.
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
