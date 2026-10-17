import collections
import glob
import json
import math
import os
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# An agent on this blog: Qwen2.5 (1.5B or 3B) in a loop with two tools,
# search_posts and read_section, until it answers or runs out of turns.
# The model never runs anything: it writes a tool call as JSON, the loop
# runs it and appends the result, and the model is called again.
# Compared with one retrieval and one answer, on 30 questions each answered
# by one known section (questions.json). Scored three ways: did a search
# return the right post, does the answer cite any source, and does it cite
# the right post. Posts that discuss these questions are left out.

# --- search: the hybrid of "Semantic and Hybrid Search" (BM25 and MiniLM, fused by rank) ---
EXCLUDE = {"function-calling-to-agent-loop", "agentic-graph-and-vectorless-rag", "recursive-language-models"}
SKIP = {"table of contents", "takeaways", "references", "where this leads"}
secs = []
for path in sorted(glob.glob("content/posts/*.mdx")):
    slug = os.path.basename(path)[:-4]
    if slug in EXCLUDE:
        continue
    text = re.sub(r"```.*?```|\$\$.*?\$\$", " ", open(path).read().split("---", 2)[-1], flags=re.S)
    parts = re.split(r"^## (.+)$", re.sub(r"<[^>]+>|^import .*$", " ", text, flags=re.M), flags=re.M)
    for h, body in zip(parts[1::2], parts[2::2]):
        h = re.sub(r"[*_`]", "", h).strip()
        if h.lower() not in SKIP and len(body.strip()) >= 200:
            secs.append({"post": slug, "heading": h, "body": " ".join(body.split())})
words = lambda t: re.findall(r"\w+", t.lower())
bags = [collections.Counter(words(s["heading"] + " " + s["body"])) for s in secs]
lens = np.array([sum(b.values()) for b in bags]); df = collections.Counter(w for b in bags for w in b)
def bm25(q, k1=1.5, b=0.75):
    s = np.zeros(len(secs))
    for w in set(words(q)) & df.keys():
        tf = np.array([d[w] for d in bags]); idf = math.log((len(secs) - df[w] + 0.5) / (df[w] + 0.5) + 1)
        s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lens / lens.mean()))
    return s
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", device="cpu")
owner, passages = zip(*[(i, " ".join(w[j:j + 80])) for i, s in enumerate(secs)
                        for w in [s["body"].split()] for j in range(0, len(w), 80)])
E = st.encode(list(passages), batch_size=64, normalize_embeddings=True); owner = np.array(owner)
def dense(q):
    best = np.full(len(secs), -np.inf); np.maximum.at(best, owner, E @ st.encode([q], normalize_embeddings=True)[0])
    return best
def hybrid(q, k=60):
    return sum(1 / (k + np.argsort(np.argsort(-f(q))) + 1) for f in (bm25, dense))

# --- the tools ---
SEEN = set()                                        # posts any search returned, for scoring
def search_posts(query):
    top = [{"post": secs[i]["post"], "heading": secs[i]["heading"], "snippet": " ".join(secs[i]["body"].split()[:60])}
           for i in np.argsort(-hybrid(query))[:3]]
    SEEN.update(t["post"] for t in top)
    return top
def read_section(post, heading):
    for s in secs:
        if s["post"] == post and s["heading"].lower() == heading.lower():
            return " ".join(s["body"].split()[:400])
    return "error: no such section; use a post and heading exactly as search_posts returned them"
RUN = {"search_posts": search_posts, "read_section": read_section}
TOOLS = [
    {"type": "function", "function": {"name": "search_posts", "description": "Search the blog. Returns the 3 best "
     "matching sections: post, heading, and the start of the text.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "read_section", "description": "Read one section in full.",
     "parameters": {"type": "object", "properties": {"post": {"type": "string"}, "heading": {"type": "string"}},
                    "required": ["post", "heading"]}}}]
SYSTEM = ("You answer questions about a technical blog, using its tools. When you know the answer, reply "
          "without a tool call: the answer in one or two sentences, then 'Source: <post>'.")
STRICT = SYSTEM + (" Always call search_posts before answering, never answer from memory, and give as the "
                   "source the post name a tool returned.")

# --- the model and the loop ---
import sys
from huggingface_hub import snapshot_download
PATH = {"1.5B": snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                                  revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
        "3B": snapshot_download("Qwen/Qwen2.5-3B-Instruct",
                                revision="aa8e72537993ba99e69dfaafa59ed015b17504d1")}[sys.argv[1] if len(sys.argv) > 1 else "1.5B"]
tok = AutoTokenizer.from_pretrained(PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
lm = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16).to(DEVICE).eval()
@torch.no_grad()
def generate(messages, tools=TOOLS, force_call=False):
    """force_call starts the reply with "<tool_call>", approximating an API's
    tool_choice = "required": the model must start a tool call."""
    prompt = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
    prefix = "<tool_call>\n" if force_call else ""
    x = tok(prompt + prefix, return_tensors="pt").input_ids.to(DEVICE)
    out = lm.generate(x, max_new_tokens=200, do_sample=False, temperature=None, top_p=None, top_k=None)
    return prefix + tok.decode(out[0, x.shape[1]:], skip_special_tokens=True), x.shape[1]

def agent(question, system=SYSTEM, force_first=False, max_turns=6):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    read, bad, calls_made = [], 0, 0
    for turn in range(1, max_turns + 1):
        text, n = generate(messages, force_call=force_first and turn == 1); read.append(n)
        calls = re.findall(r"<tool_call>\s*(\{.*?)\s*</tool_call>", text, flags=re.S)
        if not calls:                                   # no call: the model says it's done
            return text, read, bad, calls_made
        messages.append({"role": "assistant", "content": text})
        for c in calls:                                 # the loop runs the tools, not the model
            try:
                call = json.JSONDecoder().raw_decode(c)[0]; result = RUN[call["name"]](**call["arguments"])
            except Exception as e:
                bad += 1; result = f"error: {type(e).__name__}: {e}"
            calls_made += 1
            messages.append({"role": "tool", "content": json.dumps(result)})
    return "(ran out of turns)", read, bad, calls_made

def one_shot(question):
    """No loop: retrieve once with the question itself, answer once."""
    text, n = generate([{"role": "system", "content": SYSTEM}, {"role": "user", "content":
                        f"Search results: {json.dumps(search_posts(question))}\n\nQuestion: {question}"}], tools=None)
    return text, [n], 0, 1

items = json.load(open("questions.json"))
print(f"{len(secs)} sections; {len(items)} questions; Qwen2.5-{sys.argv[1] if len(sys.argv) > 1 else '1.5B'}-Instruct, greedy\n")
print(f"{'':32s} {'found':>6s} {'cites any':>10s} {'cites right':>12s} {'model calls':>12s} {'tool calls':>11s} "
      f"{'last context':>13s} {'tokens read':>12s} {'no tool':>8s} {'malformed':>10s}")
for name, fn in (("one retrieval, one answer", one_shot), ("agent loop", agent),
                 ("agent, 'always search first'", lambda q: agent(q, STRICT)),
                 ("agent, first call forced", lambda q: agent(q, STRICT, force_first=True))):
    ok, found, cites, ncalls, ntools, last, total, none, bad = [], [], [], [], [], [], [], 0, 0
    for it in items:
        SEEN.clear()
        text, read, b, tools_used = fn(it["question"])
        found.append(it["post"] in SEEN); cites.append("Source:" in text)
        with open(f"agent_log_{sys.argv[1] if len(sys.argv) > 1 else '1.5B'}.jsonl", "a") as f:        # every answer, for checking by hand
            f.write(json.dumps({"method": name, "post": it["post"], "answer": text, "calls": len(read), "found": it["post"] in SEEN,
                                "tools": tools_used, "malformed": b}) + "\n")
        ok.append(it["post"] in text.split("Source:")[-1]); ncalls.append(len(read)); ntools.append(tools_used)
        last.append(read[-1]); total.append(sum(read)); bad += b; none += fn is not one_shot and tools_used == 0
    print(f"{name:32s} {sum(found):>6d} {sum(cites):>10d} {sum(ok):>12d} {np.mean(ncalls):>12.1f} {np.mean(ntools):>11.1f} "
          f"{np.mean(last):>13,.0f} {np.mean(total):>12,.0f} {none:>8d} {bad:>10d}", flush=True)
