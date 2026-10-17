import collections
import glob
import math
import os
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# Finding the right section of this blog for a question.
#   keyword (BM25), dense (MiniLM, best 80-word passage), hybrid (both,
#   fused by reciprocal rank), each ranking every section of every post;
#   vectorless: an LLM (Qwen2.5-3B-Instruct) picks from lists:
#     titles: the list of post titles, then that post's headings;
#     +descriptions: the same, each title followed by its description;
#     3 levels: a topic (listed with its posts' titles), a post within it
#       (title and description), then a section;
#   each in two modes: reply with a number only, or reason briefly first
#   and end with "Answer: <number>". Replies without a valid number count
#   as wrong and are reported.
# 30 questions written by hand, each answered by one known section, mostly
# worded to avoid that section's heading. This post and the next are left
# out of the corpus, since they discuss these questions.
SKIP = {"table of contents", "takeaways", "references", "where this leads"}
secs, tocs, descs, topic = [], {}, {}, {}
EXCLUDE = {"agentic-graph-and-vectorless-rag", "recursive-language-models"}
for path in sorted(glob.glob("content/posts/*.mdx")):
    if os.path.basename(path)[:-4] in EXCLUDE:
        continue
    raw = open(path).read()
    slug, title = os.path.basename(path)[:-4], re.search(r'^title: "?(.*?)"?$', raw, re.M).group(1)
    text = re.sub(r"```.*?```|\$\$.*?\$\$", " ", raw.split("---", 2)[-1], flags=re.S)
    text = re.sub(r"<[^>]+>|^import .*$", " ", text, flags=re.M)
    descs[slug] = re.search(r'^description: "?(.*?)"?$', raw, re.M).group(1)
    topic[slug] = re.search(r'^topic: "?(.*?)"?$', raw, re.M).group(1)
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    for h, body in zip(parts[1::2], parts[2::2]):
        h = re.sub(r"[*_`]", "", h).strip()
        if h.lower() in SKIP or len(body.strip()) < 200:
            continue
        secs.append({"post": slug, "title": title, "heading": h, "body": " ".join(body.split())})
        tocs.setdefault(slug, (title, []))[1].append(h)

Q = [("Which of the three attention projections is never reused for later tokens, and why?", "kv-cache-explained", "Why keys and values, but not queries?"),
     ("How many bytes does the stored state take per token for a given model shape?", "kv-cache-explained", "What it costs: the memory formula"),
     ("Why is producing the first output token different from producing the rest?", "kv-cache-explained", "Two phases: prefill and decode"),
     ("How can a maximum and a running sum be updated incrementally as new scores arrive?", "flashattention-explained", "The online softmax"),
     ("Why does the gradient computation redo forward work instead of saving the score matrix?", "flashattention-explained", "The backward pass: recompute instead of store"),
     ("What limits the speed of the plain attention implementation on a GPU?", "flashattention-explained", "Where standard attention's time goes"),
     ("How does a linear penalty on token distance replace position embeddings?", "rope-and-positional-encoding", "Penalizing distance: ALiBi"),
     ("Why can't a model without order information tell 'dog bites man' from 'man bites dog'?", "rope-and-positional-encoding", "Attention can't see order"),
     ("How are the rotation speeds stretched to handle sequences longer than in training?", "rope-and-positional-encoding", "The frequencies, and longer contexts"),
     ("Why does the tokenizer work on 256 base symbols rather than Unicode characters?", "bpe-tokenizer-explained", "Start from bytes, not characters"),
     ("Why does the saved vocabulary show a strange G with a dot where spaces should be?", "bpe-tokenizer-explained", "Writing it to disk: why the vocab file is full of Ġ"),
     ("When turning new text into ids, in what order are the learned pair joins applied?", "bpe-tokenizer-explained", "Encoding: replay the merges by rank, not by count"),
     ("How should markers like end-of-text be handled so they never join with neighbors?", "bpe-tokenizer-explained", "Special tokens are boundaries, not text"),
     ("How is it prevented that a few experts receive almost all the tokens?", "mixture-of-experts", "Keeping the load even"),
     ("Why does serving a sparse expert model still need memory for every parameter?", "mixture-of-experts", "What experts cost at serving time"),
     ("What decides which experts process a given token?", "mixture-of-experts", "The router"),
     ("How does the strength of the penalty for drifting from the reference model change training?", "dpo-direct-preference-optimization", "What β does"),
     ("How does the preference loss follow from maximizing reward under a KL penalty?", "dpo-direct-preference-optimization", "The derivation"),
     ("How do compute and memory compare with the reinforcement learning pipeline it replaces?", "dpo-direct-preference-optimization", "What it costs, against PPO"),
     ("How does a method adjust the remaining columns to absorb each column's rounding error?", "quantization-from-scratch", "GPTQ: rounding with error compensation"),
     ("Should one quantization grid be shared by a whole tensor, each row, or small groups?", "quantization-from-scratch", "Where to put the scales"),
     ("Why does storing weights in 4 bits make token generation faster?", "quantization-from-scratch", "What it buys at decode"),
     ("Why does checking several proposed tokens cost about the same as generating one?", "speculative-decoding", "Why verification is nearly free"),
     ("How is the output kept exactly distributed as the large model's despite a cheap guesser?", "speculative-decoding", "The acceptance rule, and why it is exact"),
     ("In which serving conditions does guessing ahead stop paying off?", "speculative-decoding", "When it doesn't help"),
     ("How do two sequences share the same prompt memory until one of them writes to it?", "paged-attention-explained", "Memory sharing: copy-on-write"),
     ("How much memory do contiguous reservations waste on unused slots?", "paged-attention-explained", "The problem: reserved but empty"),
     ("What maps a sequence's logical positions to physical memory chunks?", "paged-attention-explained", "The block table"),
     ("How are two ranked lists combined into one without calibrating their scores?", "semantic-and-hybrid-search", "Fusing rankings"),
     ("How is term frequency saturated and document length normalized in keyword scoring?", "semantic-and-hybrid-search", "BM25: keyword search, derived")]
index = {(s["post"], s["heading"]): i for i, s in enumerate(secs)}
targets = [index[(p, h)] for _, p, h in Q]

tokens = lambda t: re.findall(r"\w+", t.lower())
class BM25:
    """Okapi BM25, k1 = 1.5, b = 0.75 (see "Semantic and Hybrid Search")."""
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [collections.Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(d.values()) for d in self.docs]); self.avg = self.len.mean()
        df = collections.Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log((len(docs) - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
        self.k1, self.b = k1, b
    def scores(self, q):
        s = np.zeros(len(self.docs))
        for t in set(tokens(q)):
            if t in self.idf:
                tf = np.array([d[t] for d in self.docs])
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len / self.avg))
        return s
docs = [s["title"] + ". " + s["heading"] + ". " + s["body"] for s in secs]
bm25 = BM25(docs).scores
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", device="cpu")
owner, passages = [], []
for i, d in enumerate(docs):
    w = d.split()
    for j in range(0, len(w), 80):
        owner.append(i); passages.append(" ".join(w[j:j + 80]))
P = st.encode(passages, batch_size=64, normalize_embeddings=True); owner = np.array(owner)
def dense(q):
    s = P @ st.encode([q], normalize_embeddings=True)[0]
    best = np.full(len(docs), -np.inf); np.maximum.at(best, owner, s)
    return best
def rrf(q, k=60):
    total = 0
    for fn in (bm25, dense):
        order = np.argsort(-fn(q)); rank = np.empty_like(order); rank[order] = np.arange(1, len(order) + 1)
        total = total + 1 / (k + rank)
    return total

PATH = snapshot_download("Qwen/Qwen2.5-3B-Instruct",
                         revision="aa8e72537993ba99e69dfaafa59ed015b17504d1")
tok = AutoTokenizer.from_pretrained(PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
lm = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16).to(DEVICE).eval()
stats = collections.Counter()                     # calls, invalid replies, prompt tokens
@torch.no_grad()
def pick(prompt, options, reason):
    """The LLM picks an option by number; None if the reply has no valid number."""
    listing = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
    ask = ("Think briefly about which option fits, then end with: Answer: <number>" if reason
           else "Reply with the number only.")
    ids = tok.apply_chat_template([{"role": "user", "content": f"{prompt}\n\n{listing}\n\n{ask}"}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(DEVICE)
    out = lm.generate(ids, max_new_tokens=300 if reason else 6, do_sample=False, temperature=None, top_p=None, top_k=None)
    reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"Answer:\s*\**\s*(\d+)", reply) if reason else re.search(r"\d+", reply)
    k = int(m.group(1 if reason else 0)) - 1 if m else -1
    stats["calls"] += 1; stats["prompt"] += ids.shape[1]
    if not 0 <= k < len(options):
        stats["invalid"] += 1
        return None
    return k

slugs = list(tocs)
topics = sorted(set(topic[s] for s in slugs))
def section(q, slug, reason):
    h = pick(f"Question: {q}\nWhich section of the post '{tocs[slug][0]}' answers it?", tocs[slug][1], reason)
    return None if h is None else index[(slug, tocs[slug][1][h])]
def flat(q, reason, described=False, order=None):
    order = order or slugs
    p = pick(f"Question: {q}\nWhich blog post most likely answers it?",
             [tocs[s][0] + (f": {descs[s]}" if described else "") for s in order], reason)
    return (None, None, p) if p is None else (order[p], section(q, order[p], reason), p)
def three_levels(q, reason):
    t = pick(f"Question: {q}\nWhich topic's posts most likely answer it?",
             [f"{t}: " + "; ".join(tocs[s][0] for s in slugs if topic[s] == t) for t in topics], reason)
    if t is None:
        return None, None, None
    in_t = [s for s in slugs if topic[s] == topics[t]]
    p = pick(f"Question: {q}\nWhich blog post most likely answers it?", [f"{tocs[s][0]}: {descs[s]}" for s in in_t], reason)
    return (None, None, topics[t]) if p is None else (in_t[p], section(q, in_t[p], reason), topics[t])

print(f"{len(secs)} sections in {len(tocs)} posts; {len(Q)} questions; counts are out of {len(Q)}\n")
print(f"{'method':42s} {'right post':>10s} {'right section':>14s} {'section in top 5':>17s}")
ranked = {}
for name, fn in (("keyword (BM25)", bm25), ("dense (MiniLM)", dense), ("hybrid (fused)", rrf)):
    ranks = ranked[name] = [np.argsort(-fn(q))[:5] for q, _, _ in Q]
    print(f"{name:42s} {sum(secs[r[0]]['post'] == p for r, (_, p, _) in zip(ranks, Q)):>10d} "
          f"{sum(r[0] == t for r, t in zip(ranks, targets)):>14d} {sum(t in r for r, t in zip(ranks, targets)):>17d}", flush=True)

def fifths(picks, n):
    return "/".join(str(sum(p is not None and int(5 * p / n) == f for p in picks)) for f in range(5))
for reason in (False, True):
    mode = "reasoning" if reason else "number only"
    for label, run in (("titles", lambda q: flat(q, reason)), ("+descriptions", lambda q: flat(q, reason, True)),
                       ("3 levels", lambda q: three_levels(q, reason))):
        stats.clear()
        res = [run(q) for q, _, _ in Q]
        print(f"{'vectorless, ' + label + ', ' + mode:42s} {sum(r[0] == p for r, (_, p, _) in zip(res, Q)):>10d} "
              f"{sum(r[1] == t for r, t in zip(res, targets)):>14d} {'-':>17s}", flush=True)
        extra = ""
        if label == "titles":
            extra = f"; picks by fifth of the list {fifths([r[2] for r in res], len(slugs))}"
        if label == "3 levels":
            ok_t = [r[2] == topic[p] for r, (_, p, _) in zip(res, Q)]
            extra = (f"; right topic {sum(ok_t)} (of {len(topics)} topics), then right post "
                     f"{sum(r[0] == p for r, (_, p, _), o in zip(res, Q, ok_t) if o)} of those")
        print(f"   {stats['calls']} calls, {stats['prompt'] / len(Q):,.0f} prompt tokens per question, "
              f"{stats['invalid']} invalid replies{extra}", flush=True)
    stats.clear()
    rev = [flat(q, reason, order=slugs[::-1]) for q, _, _ in Q]
    print(f"   titles reversed: right post {sum(r[0] == p for r, (_, p, _) in zip(rev, Q))}; "
          f"picks by fifth of the reversed list {fifths([r[2] for r in rev], len(slugs))}", flush=True)
    stats.clear()
    sec = [section(q, p, reason) for q, p, _ in Q]
    print(f"   section only, right post given: {sum(s == t for s, t in zip(sec, targets))} "
          f"(chance {sum(1 / len(tocs[p][1]) for _, p, _ in Q):.1f}); {stats['invalid']} invalid replies", flush=True)

# Where keyword search beat dense search: the question's words found in the right section.
idf = BM25(docs).idf
print("\nquestions where BM25 found the right post and dense search didn't:")
for (q, p, h), rb, rd, t in zip(Q, ranked["keyword (BM25)"], ranked["dense (MiniLM)"], targets):
    if secs[rb[0]]["post"] == p and secs[rd[0]]["post"] != p:
        shared = sorted(set(tokens(q)) & set(tokens(docs[t])), key=lambda w: -idf.get(w, 0))[:3]
        print(f"   {q[:62]!r}: rarest shared words {shared}")