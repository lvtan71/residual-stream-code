import numpy as np
from router_data import split, tokens, ROUTES
from softmax_basics import softmax
from router4 import four_way, train

tr, va, te = split(0)
((Xtr, ytr), (Xte, yte)), vocab = four_way(tr, te)
W, b = train(Xtr, ytr, smooth=0.1)
for k, route in enumerate(ROUTES):
    top = np.argsort(-W[:, k])[:6]
    print(f"{route:13s} " + ", ".join(f"{vocab[j]} {W[j, k]:+.2f}" for j in top))
print(f"\nbias b = {np.round(b, 2)}, softmax(b) = {np.round(softmax(b), 2)}")
P = softmax(Xte @ W + b)
for i in np.where((yte == 1) & (P.argmax(1) == 3))[0]:
    words = tokens(te[i]["question"])
    print(f"\narchitecture -> foundations: {te[i]['question']}")
    print("  never seen in training: " + ", ".join(t for t in words if t not in vocab))
    for t in dict.fromkeys(t for t in words if t in vocab):
        print(f"  {t:5s} weights {np.round(W[vocab.index(t)], 2)}")
    print(f"  words alone: p = {np.round(softmax(Xte[i] @ W), 2)}   with the bias: p = {np.round(P[i], 2)}")