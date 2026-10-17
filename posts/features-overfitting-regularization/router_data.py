import csv
import re
import numpy as np

ROUTES = ["serving", "architecture", "agents", "foundations"]
rows = list(csv.DictReader(open("router-questions.csv")))
english = [r for r in rows if r["lang"] == "en"]
vietnamese = [r for r in rows if r["lang"] == "vi"]

def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def split(seed):
    """30 train / 10 validation / 10 test questions per route."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for route in ROUTES:
        group = [r for r in english if r["route"] == route]
        for k, i in enumerate(rng.permutation(len(group))):
            (train if k < 30 else val if k < 40 else test).append(group[i])
    return train, val, test

def featurize(train, *others):
    """Bag of words: one 0/1 column per word seen in *training* questions."""
    vocab = sorted({t for r in train for t in tokens(r["question"])})
    column = {t: j for j, t in enumerate(vocab)}
    out = []
    for rs in (train, *others):
        X = np.zeros((len(rs), len(vocab)))
        for i, r in enumerate(rs):
            for t in set(tokens(r["question"])):
                if t in column:          # words never seen in training are dropped
                    X[i, column[t]] = 1
        y = np.array([r["route"] == "serving" for r in rs], dtype=float)
        out.append((X, y))
    return out, vocab