import hashlib
import json
import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from common import posts

# Retrieval for RAG on this blog, measured. 150 questions (gen.json), each
# answered by one sentence of a post; a retrieval hits if a returned chunk
# contains the whole sentence. Dense retrieval with MiniLM ("Semantic and Hybrid Search"), top 1
# and top 5, under eight ways of cutting the posts into chunks and one control; paired
# comparisons use an exact sign test on the questions where two disagree. Then HyDE,
# an embedding cache and a semantic cache.
items = json.load(open("gen.json"))
docs = posts()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", device="cpu")
norm = lambda t: " ".join(t.replace(">", " ").split())        # whitespace and quote marks don't count

# --- the embedding cache: key = hash(model name, text) ---
CACHE, MODEL = {}, "all-MiniLM-L6-v2"
def embed(texts):
    keys = [hashlib.sha256(f"{MODEL}\0{t}".encode()).hexdigest() for t in texts]
    todo = [t for t, k in zip(texts, keys) if k not in CACHE]
    for t, v in zip(todo, model.encode(todo, batch_size=64, normalize_embeddings=True) if todo else []):
        CACHE[hashlib.sha256(f"{MODEL}\0{t}".encode()).hexdigest()] = v
    return np.array([CACHE[k] for k in keys])

# --- the chunkers; each returns (text to embed, text returned to the model) ---
def sentences(text):                                       # the same rule that cut the gold sentences
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s]
def sections(slug):
    title, text = docs[slug]
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    return [("", parts[0])] + list(zip(parts[1::2], parts[2::2]))
def windows(words, size, stride):
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - size + stride, 1), stride)]
def recursive(text, size=200):
    """Paragraphs merged up to `size` words; a longer paragraph split by sentences."""
    out, cur = [], []
    for para in re.split(r"\n\s*\n", text):
        units = [para] if len(para.split()) <= size else sentences(para)
        for u in units:
            if cur and len(" ".join(cur).split()) + len(u.split()) > size:
                out.append(" ".join(cur)); cur = []
            cur.append(u)
    return out + ([" ".join(cur)] if cur else [])
def semantic(text, size=200):
    """Sentences grouped; a new chunk starts where adjacent sentences are least similar."""
    sents = sentences(" ".join(text.split()))
    if len(sents) < 2:
        return [" ".join(sents)]
    e = embed(sents); sim = (e[:-1] * e[1:]).sum(1); cut = np.quantile(sim, 0.2)
    out, cur = [], [sents[0]]
    for s, v in zip(sents[1:], sim):
        if v < cut or len(" ".join(cur + [s]).split()) > size:
            out.append(" ".join(cur)); cur = []
        cur.append(s)
    return out + [" ".join(cur)]

def chunk(strategy):
    out = []
    for slug, (title, text) in docs.items():
        if strategy == "fixed, 200 words":
            out += [(c, c) for c in windows(text.split(), 200, 200)]
        elif strategy == "fixed, 200 words, 50 overlap":
            out += [(c, c) for c in windows(text.split(), 200, 150)]
        elif strategy == "recursive, up to 200 words":
            out += [(c, c) for c in recursive(text)]
        elif strategy == "recursive, up to 100 words":
            out += [(c, c) for c in recursive(text, 100)]
        elif strategy == "one chunk per section":
            out += [(f"{h}\n{b}", f"{h}\n{b}") for h, b in sections(slug) if b.strip()]
        elif strategy == "semantic, up to 200 words":
            out += [(c, c) for c in semantic(text)]
        elif strategy == "recursive within each section":
            out += [(c, c) for h, b in sections(slug) for c in recursive(b)]
        elif strategy == "contextual: title + heading added":
            out += [(f"{title}. {h}. {c}", c) for h, b in sections(slug) for c in recursive(b)]
        elif strategy == "small-to-big: 60 words -> section":
            out += [(c, f"{h}\n{b}") for h, b in sections(slug) for c in windows(b.split(), 60, 60)]
    return out

def hits(queries, chunks, E, k):
    found = np.argsort(-(embed(queries) @ E.T), axis=1)[:, :k]
    gold = [norm(it["sentence"]) for it in items]
    return np.array([any(g in norm(chunks[j][1]) for j in row) for g, row in zip(gold, found)]), found

STRATEGIES = ["fixed, 200 words", "fixed, 200 words, 50 overlap", "recursive, up to 200 words", "recursive, up to 100 words",
              "one chunk per section", "semantic, up to 200 words", "recursive within each section",
              "contextual: title + heading added",
              "small-to-big: 60 words -> section"]
questions = [it["question"] for it in items]
print(f"{len(docs)} posts, {len(items)} questions; MiniLM, dense retrieval\n")
from math import comb
def sign_test(a, b):
    """Two-sided exact sign test on the questions where a and b disagree."""
    x, y = int((a & ~b).sum()), int((b & ~a).sum()); n = x + y
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return f"{x} vs {y}, p = {p:.2f}"
print(f"{'chunking':36s} {'chunks':>7s} {'words each':>11s} {'findable':>9s} {'hit@1':>6s} {'hit@5':>6s} "
      f"{'words returned (top 5)':>23s}  top 5 against recursive 200")
built, top5 = {}, {}
for s in STRATEGIES:
    chunks = chunk(s); E = embed([c[0] for c in chunks]); built[s] = (chunks, E)
    findable = sum(any(norm(it["sentence"]) in norm(c[1]) for c in chunks) for it in items)
    h1, _ = hits(questions, chunks, E, 1); h5, top = hits(questions, chunks, E, 5); top5[s] = h5
    returned = np.mean([len(" ".join(dict.fromkeys(chunks[j][1] for j in row)).split()) for row in top])
    vs = "" if s == "recursive, up to 200 words" or "recursive, up to 200 words" not in top5 else sign_test(h5, top5["recursive, up to 200 words"])
    print(f"{s:36s} {len(chunks):>7,d} {np.mean([len(c[1].split()) for c in chunks]):>11.0f} {findable:>9d} "
          f"{h1.sum():>4d}/{len(items)} {h5.sum():>3d}/{len(items)} {returned:>23,.0f}  {vs}", flush=True)
print(f"fixed windows, overlap against none: {sign_test(top5['fixed, 200 words, 50 overlap'], top5['fixed, 200 words'])}; "
      f"recursive 200 against fixed: {sign_test(top5['recursive, up to 200 words'], top5['fixed, 200 words'])}")
chunks, E = built["recursive, up to 100 words"]
h10, top = hits(questions, chunks, E, 10)
print(f"recursive 100 at top 10 (a word budget near small-to-big's): hit {h10.sum()}/{len(items)}, "
      f"{np.mean([len(' '.join(chunks[j][1] for j in row).split()) for row in top]):,.0f} words returned")
print("(MiniLM reads at most 256 word pieces, about 170 words of this blog; the rest of a longer chunk is not embedded)")

# --- HyDE: embed a hypothetical answer instead of the question ---
chunks, E = built["recursive, up to 200 words"]
print("\nHyDE, on recursive chunks: what is embedded as the query")
base5, _ = hits(questions, chunks, E, 5)
def hits_vec(V, chunks, E, k):                             # hits for query vectors given directly
    found = np.argsort(-(V @ E.T), axis=1)[:, :k]
    return np.array([any(norm(it["sentence"]) in norm(chunks[j][1]) for j in row) for it, row in zip(items, found)])
avg = embed(questions) + embed([it["hyde"] for it in items])            # the paper's way: average the two embeddings
avg /= np.linalg.norm(avg, axis=1, keepdims=True)
for name, qs in (("the question", questions), ("a hypothetical answer (HyDE)", [it["hyde"] for it in items]),
                 ("question + answer, texts joined", [f"{it['question']} {it['hyde']}" for it in items]),
                 ("question and answer embeddings averaged", avg)):
    if isinstance(qs, np.ndarray):
        h1, h5 = hits_vec(qs, chunks, E, 1), hits_vec(qs, chunks, E, 5)
    else:
        h1, _ = hits(qs, chunks, E, 1); h5, _ = hits(qs, chunks, E, 5)
    print(f"  {name:32s} hit@1 {h1.sum():>3d}  hit@5 {h5.sum():>3d}  "
          f"(vs the question at top 5: {int((h5 & ~base5).sum())} gained, {int((base5 & ~h5).sum())} lost)")

# --- the embedding cache ---
CACHE.clear(); t0 = time.perf_counter(); embed([c[0] for c in chunks]); cold = time.perf_counter() - t0
t0 = time.perf_counter(); embed([c[0] for c in chunks]); warm = time.perf_counter() - t0
print(f"\nembedding cache: {len(chunks):,} chunks embedded in {cold:.1f} s cold, {warm:.3f} s from the cache")

# --- the semantic cache: answer a query from a cached one if they're similar enough ---
# The first 75 questions are in the cache; the queries are rewordings of all
# 150, so half are repeats (the right hit is their own original) and half are new.
cached = embed(questions[:75])
queries, own = [], []
for i, it in enumerate(items):
    for r in it["reworded"]:
        queries.append(r); own.append(i if i < 75 else -1)
own = np.array(own); S = embed(queries) @ cached.T; best, score = S.argmax(1), S.max(1)
print("\nsemantic cache: 75 questions cached; 300 queries, 150 of them rewordings of a cached question")
print(f"{'threshold':>10s} {'repeats served':>15s} {'wrong answers served':>21s} {'of which to new queries':>24s}")
for t in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
    served = score >= t
    right = served & (best == own); wrong = served & (best != own)
    print(f"{t:>10.2f} {right.sum():>11d}/150 {wrong.sum():>17d}/300 {(wrong & (own < 0)).sum():>20d}/150")

# A harder test: different questions about the same post. They're the
# realistic near-misses: similar wording, different answers.
Q = embed(questions); posts_of = [it["post"] for it in items]
pairs = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items)) if posts_of[i] == posts_of[j]]
own = embed([it["reworded"][0] for it in items]) @ Q.T
print(f"\nharder test: {len(pairs)} pairs of different questions about the same post, against each question's own rewording")
print(f"{'threshold':>10s} {'same-post pairs above it (would be confused)':>45s} {'rewordings above it (hits)':>28s}")
for t in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
    confused = sum(Q[i] @ Q[j] >= t for i, j in pairs)
    print(f"{t:>10.2f} {confused:>37d}/{len(pairs)} {sum(own[i, i] >= t for i in range(len(items))):>24d}/{len(items)}")
print(f"  highest similarity between two different same-post questions: {max(Q[i] @ Q[j] for i, j in pairs):.2f}")

# The hardest case, written by hand: questions that differ by a word or two
# and need different answers. A cache that serves one for the other is wrong.
NEAR = [("What does forward KL do when the teacher has two modes?", "What does reverse KL do when the teacher has two modes?"),
        ("What does LoRA freeze during fine-tuning?", "What does LoRA train during fine-tuning?"),
        ("Why is decode memory-bound?", "Why is prefill compute-bound?"),
        ("Which serving engine is NVIDIA-only?", "Which serving engine supports Huawei Ascend?"),
        ("What does BM25's k1 parameter control?", "What does BM25's b parameter control?"),
        ("Who introduced iteration-level scheduling?", "Who introduced PagedAttention?"),
        ("What is the derivative of the sigmoid?", "What is the derivative of tanh?"),
        ("When does speculative decoding help?", "When does speculative decoding not help?"),
        ("How many bytes per parameter does training with AdamW need?", "How many bytes per parameter does inference in FP16 need?"),
        ("What is vLLM's default token budget for chunked prefill?", "What is vLLM's default KV-cache block size?"),
        ("Why do keys and values get cached but not queries?", "Why do queries get cached but not keys and values?"),
        ("How does GPTQ choose its rounding?", "How does AWQ choose its scales?")]
sims = [float(embed([a])[0] @ embed([b])[0]) for a, b in NEAR]
print(f"\nhardest test: {len(NEAR)} hand-written pairs that differ by a word or two and need different answers")
print("  similarities: " + ", ".join(f"{s:.2f}" for s in sorted(sims, reverse=True)))
for t in (0.80, 0.85, 0.90, 0.95):
    print(f"  at threshold {t:.2f}: {sum(s >= t for s in sims)} of {len(NEAR)} pairs would share one cached answer")