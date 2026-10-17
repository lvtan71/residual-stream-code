import numpy as np
from router_data import english, split, tokens, ROUTES
from softmax_basics import softmax
from router4 import four_way, train

unsure, sent_to, all_to, unseen = [], np.zeros(4), np.zeros(4), {True: [], False: []}
to_found, moved, right = 0, 0, {True: [], False: []}
for seed in range(20):
    tr, va, te = split(seed)
    ((Xtr, ytr), (Xte, yte)), vocab = four_way(tr, te)
    W, b = train(Xtr, ytr, smooth=0.1)
    P = softmax(Xte @ W + b)
    low = P.max(1) < 0.4                                 # the router is barely guessing
    unsure.append(low.mean())
    sent_to += np.bincount(P[low].argmax(1), minlength=4)
    all_to += np.bincount(P.argmax(1), minlength=4)
    for i, r in enumerate(te):
        unseen[bool(low[i])].append(np.mean([t not in vocab for t in tokens(r["question"])]))
        right[bool(low[i])].append(P[i].argmax() == yte[i])
    # Would they still go to foundations if no route had a head start from its bias?
    found = low & (P.argmax(1) == 3)
    to_found += found.sum()
    moved += (softmax(Xte[found] @ W + b.mean()).argmax(1) != 3).sum()
print(f"test questions with top probability < 0.4: {np.mean(unsure):.0%}, "
      f"right {np.mean(right[True]):.0%} of the time (the rest: {np.mean(right[False]):.0%})")
print("where they go:   " + ", ".join(f"{r} {n / sent_to.sum():.0%}" for r, n in zip(ROUTES, sent_to)))
print("all predictions: " + ", ".join(f"{r} {n / all_to.sum():.0%}" for r, n in zip(ROUTES, all_to)))
print(f"share of words never seen in training: unsure {np.mean(unseen[True]):.0%}, "
      f"the rest {np.mean(unseen[False]):.0%}")
print(f"of the {to_found} unsure questions sent to foundations, {moved} go elsewhere with an even bias")
for w in ["what", "is"]:
    print(f"questions containing '{w}': " + ", ".join(
        f"{r} {sum(w in tokens(q['question']) for q in english if q['route'] == r)}" for r in ROUTES))