import re
import time
import quiet_machine                                   # waits for an idle machine before anything is timed
import numpy as np
import torch
from router_data import split, ROUTES, english, tokens
from router4 import four_way, train, top_label_ece
from softmax_basics import softmax

# Ways to make one typed decision, which of four routes a question belongs
# to, on the series' 200 English router questions. Every question gets one
# out-of-sample prediction: the trained models use the first of 40 random
# splits (the 120 / 40 / 40 split of "Features and Overfitting") in which that question is a test
# question; the language model is not trained, and answers each question
# once (with examples, they come from that same split's training set).
#   1. the bag-of-words softmax router of "Cross-Entropy and the Softmax Gradient";
#   2. sentence embeddings (all-MiniLM-L6-v2, 23M parameters, "Embeddings and Contrastive Learning") with
#      the same softmax head, and with no training at all: nearest centroid;
#   3. Qwen2.5-1.5B-Instruct, untrained, zero-shot and with 20 labeled
#      examples in the prompt, read two ways: typed, by the probability of
#      every way of spelling each route name; or free, generating and parsing.
SPLITS = [split(seed) for seed in range(40)]
home = {}                                                   # question -> index of its test split
for k, (_, _, test) in enumerate(SPLITS):
    for r in test:
        home.setdefault(r["question"], k)
questions = [r["question"] for r in english]
truth = np.array([ROUTES.index(r["route"]) for r in english])
assert len(home) == 200
preds = {}

# 1. Bag of words, and its batch-1 latency, text in to probabilities out.
bow, t_bow = {}, []
for k, (tr, _, te) in enumerate(SPLITS):
    ((Xtr, ytr), _), vocab = four_way(tr, te)
    W, b = train(Xtr, ytr, 0.0, 0.1)                                       # the choice of "Cross-Entropy and the Softmax Gradient"
    col = {t: j for j, t in enumerate(vocab)}
    for r in te:
        if home[r["question"]] != k:
            continue
        t0 = time.perf_counter()
        x = np.zeros(len(vocab))
        for t in set(tokens(r["question"])):
            if t in col:
                x[col[t]] = 1
        p = softmax(x[None] @ W + b)[0]
        t_bow.append(time.perf_counter() - t0)
        bow[r["question"]] = p
preds["bag-of-words router"] = (bow, np.median(t_bow))

# 2. Sentence embeddings.
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", device="cpu")
E = dict(zip(questions, enc.encode(questions, normalize_embeddings=True)))
t_enc = []
for q in questions[:50]:                                                   # batch-1 encode time
    t0 = time.perf_counter(); enc.encode([q], normalize_embeddings=True); t_enc.append(time.perf_counter() - t0)
head, centroid = {}, {}
for k, (tr, _, te) in enumerate(SPLITS):
    Xtr = np.array([E[r["question"]] for r in tr]); ytr = np.array([ROUTES.index(r["route"]) for r in tr])
    W, b = train(Xtr * 8, ytr, 0.0, 0.1)                                   # x8: unit vectors train faster scaled up
    C = np.array([Xtr[ytr == c].mean(0) for c in range(4)])
    for r in te:
        if home[r["question"]] == k:
            e = E[r["question"]]
            head[r["question"]] = softmax((e * 8)[None] @ W + b)[0]
            centroid[r["question"]] = softmax((e @ C.T)[None] * 20)[0]    # cosine to each route's mean
preds["MiniLM + softmax head"] = (head, np.median(t_enc))
preds["MiniLM, nearest centroid"] = (centroid, np.median(t_enc))

# 3. The language model.
from lm import model, tok, DEVICE
sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(DEVICE, lambda: None)   # the post ran on "mps"
m = model()
DESCRIBE = ("serving: running models in production (KV cache, batching, vLLM, latency); "
            "architecture: how models are built (attention, layers, positions, MoE); "
            "agents: tools, planning and agent loops; "
            "foundations: training, losses, optimization and math.")
SPELLINGS = {r: [r, r.capitalize(), " " + r, " " + r.capitalize()] for r in ROUTES}
SPELLINGS["foundations"] += ["foundation", "Foundation", " foundation", " Foundation"]
SPELLINGS["agents"] += ["agent", "Agent", " agent", " Agent"]
cands = [(ROUTES.index(r), tok(s).input_ids) for r, forms in SPELLINGS.items() for s in forms]

def prompt(q, shots=None):
    if shots is None:
        text = (f"Classify the question into exactly one route.\nRoutes: {DESCRIBE}\nQuestion: {q}\n"
                "Answer with one word: serving, architecture, agents or foundations.")
    else:
        examples = "\n".join(f"Question: {r['question']}\nRoute: {r['route']}" for r in shots)
        text = (f"Classify the question into exactly one route.\nRoutes: {DESCRIBE}\n"
                f"Labeled examples:\n{examples}\n\nQuestion: {q}\n"
                "Answer with one word: serving, architecture, agents or foundations.")
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True)["input_ids"][0].tolist()

@torch.no_grad()
def typed(ids):
    """Probability of each route: the summed probability of every spelling's
    full token sequence after the prompt, scored in one batched forward."""
    L = max(len(c) for _, c in cands)
    seqs = [ids + c + [tok.eos_token_id] * (L - len(c)) for _, c in cands]
    h = m.model(torch.tensor(seqs, device=DEVICE)).last_hidden_state[:, len(ids) - 1:]   # only the scored positions
    logp = m.lm_head(h).float().log_softmax(-1)
    p = np.zeros(4)
    for i, (route, c) in enumerate(cands):
        p[route] += float(sum(logp[i, j, t] for j, t in enumerate(c)).exp())
    return p / p.sum(), p.sum()

def parse(text):
    """A forgiving parser: lowercase, drop punctuation and markdown, accept
    the singular and a leading prefix."""
    w = re.sub(r"[^a-z ]", " ", text.lower()).split()
    for word in w[:3]:
        for r in ROUTES:
            if word in (r, r.rstrip("s")) or (len(word) >= 5 and r.startswith(word)):
                return ROUTES.index(r)
    return None

def shots_for(q):
    tr = SPLITS[home[q]][0]
    rng = np.random.default_rng(home[q])
    pool = [tr[i] for i in rng.permutation(len(tr))]
    s = [r for route in ROUTES for r in [x for x in pool if x["route"] == route][:5]]
    rng.shuffle(s)
    return s

for name, with_shots in (("Qwen2.5-1.5B zero-shot", False), ("Qwen2.5-1.5B, 20 examples", True)):
    typed_p, free_p, covered, invalid, t_pre, t_gen, lens = {}, {}, [], 0, [], [], []
    for q in questions:
        ids = prompt(q, shots_for(q) if with_shots else None)
        lens.append(len(ids))
        x = torch.tensor([ids], device=DEVICE)
        sync(); t0 = time.perf_counter()
        with torch.no_grad():
            m(x)
        sync(); t_pre.append(time.perf_counter() - t0)
        typed_p[q], mass = typed(ids); covered.append(mass)
        sync(); t0 = time.perf_counter()
        out = m.generate(x, max_new_tokens=8, do_sample=False, repetition_penalty=1.0)
        sync(); t_gen.append(time.perf_counter() - t0)
        r = parse(tok.decode(out[0, len(ids):], skip_special_tokens=True))
        if r is None:
            invalid += 1
        free_p[q] = None if r is None else np.eye(4)[r]
    preds[f"{name}, typed"] = (typed_p, np.median(t_pre))
    preds[f"{name}, free"] = (free_p, np.median(t_gen))
    print(f"{name}: prompts of {np.mean(lens):.0f} tokens on average; the route spellings carry "
          f"{np.median(covered):.0%} of the first-word probability (median); free answers unparseable: {invalid} of 200")

# Scores, one prediction per question. 95% intervals by bootstrap over questions.
rng = np.random.default_rng(0)
boots = [rng.integers(0, 200, 200) for _ in range(2000)]
print(f"\n{'':40s} {'accuracy':>16s} {'ECE':>6s} {'batch-1 time':>13s}")
correct = {}
for name, (p, t) in preds.items():
    right = np.array([p[q] is not None and p[q].argmax() == y for q, y in zip(questions, truth)], dtype=float)
    correct[name] = right
    lo, hi = np.percentile([right[i].mean() for i in boots], [2.5, 97.5])
    if all(p[q] is not None for q in questions) and "free" not in name:
        P = np.array([p[q] for q in questions]); ece = f"{top_label_ece(P, truth):6.3f}"
    else:
        ece = "   n/a"                                                       # generated labels carry no confidence
    when = "" if t is None else (f"{t * 1e6:8.1f} us" if t < 1e-3 else f"{t * 1e3:8.1f} ms")
    print(f"{name:40s} {right.mean():.3f} [{lo:.2f}, {hi:.2f}] {ece} {when:>13s}")
d = correct["MiniLM + softmax head"] - correct["Qwen2.5-1.5B, 20 examples, typed"]
lo, hi = np.percentile([d[i].mean() for i in boots], [2.5, 97.5])
print(f"\nMiniLM head minus Qwen with examples (typed): {d.mean():+.3f} [{lo:+.2f}, {hi:+.2f}]")
P = np.array([head[q] for q in questions])
for th in (0.6, 0.7, 0.8):
    keep = P.max(1) >= th
    print(f"MiniLM head, answering only at confidence >= {th}: {keep.mean():.0%} of questions answered, "
          f"{(P[keep].argmax(1) == truth[keep]).mean():.1%} of those right")
qp = np.array([preds['Qwen2.5-1.5B, 20 examples, typed'][0][q] for q in questions])
print("Qwen with examples, typed, right by true route: " + ", ".join(
    f"{r} {np.mean(qp[truth == i].argmax(1) == i):.2f}" for i, r in enumerate(ROUTES)))
