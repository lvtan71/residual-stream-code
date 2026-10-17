import ast
import contextlib
import glob
import io
import json
import os
import random
import re
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import clean                                    # prose only: no code, math, components or URLs
from huggingface_hub import snapshot_download

# A miniature recursive language model (after Zhang, Kraska and Khattab).
# The long input never enters the model's context: it sits in a Python REPL
# as the variable `context`. The model writes code to inspect it, sees only
# what its code prints (cut to 2,000 characters), and can call
# llm_query(text) to have a fresh copy of itself read a piece. The system
# prompt is a shortened version of the paper's, with its examples. Model:
# Qwen2.5-3B-Instruct, a general instruct model, not a coding model.
# Compared with: the same model reading the input directly, cut to its first
# 8,000 tokens; and a fixed program making the same kind of sub-calls.
# Input: this blog's posts as prose, concatenated. Posts that discuss this
# experiment are left out.
PATH = snapshot_download("Qwen/Qwen2.5-3B-Instruct",
                         revision="aa8e72537993ba99e69dfaafa59ed015b17504d1")
tok = AutoTokenizer.from_pretrained(PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16).to(DEVICE).eval()

EXCLUDE = {"recursive-language-models", "agentic-graph-and-vectorless-rag", "function-calling-to-agent-loop",
           "building-rag-chunking-hyde-and-caching"}
posts = []
for p in sorted(glob.glob("content/posts/*.mdx")):
    if os.path.basename(p)[:-4] not in EXCLUDE:
        raw = open(p).read()
        title = re.search(r'^title: "?(.*?)"?$', raw, re.M).group(1)
        posts.append(f"=== POST: {title} ===\n{clean(raw)}")
rng = random.Random(0)

@torch.no_grad()
def chat(messages, max_new=400):
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)["input_ids"].to(DEVICE)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
    chat.calls += 1
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

SUB = []                                                  # every sub-call's question and reply
def llm_query(text, max_new=300):
    if len(text) > 20000:                                 # ~6k tokens: keeps memory in bounds on this laptop
        return f"error: llm_query takes at most 20,000 characters; this text has {len(text):,}"
    reply = chat([{"role": "user", "content": text}], max_new=max_new)
    SUB.append({"asked": text[-160:], "reply": reply[:160]})
    return reply

SYSTEM = """You are answering a query about a context stored in a Python REPL as the variable `context`, a string of {n:,} characters (a blog's posts, each starting with a line '=== POST: <title> ==='). You cannot see it directly. Write Python in ```python blocks; you see only what you print, cut to 2,000 characters. Variables persist between turns, so use them as buffers. The REPL also has llm_query(text), which asks another model about up to 20,000 characters of text; use it when you need to understand meaning, not just match strings. Make sure your code covers the whole context before you answer.

A good strategy is to look at the context first, decide how to split it, query an LLM per piece with a specific question, collect the answers, and combine them. For example, to find a magic number in a long context:
```python
answers = []
for i in range(0, len(context), 19000):
    answers.append(llm_query("What is the magic number in this text? Reply NONE if there is none.\\n" + context[i:i + 19000]))
print([a for a in answers if "NONE" not in a])
```
When you are done, reply with FINAL(your answer), or FINAL_VAR(name) to return a REPL variable, and nothing else."""

LOG = []
def run(code, env):
    """Execute like a REPL: the value of a final bare expression is printed."""
    tree = ast.parse(code)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, "<repl>", "exec"), env)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), "<repl>", "eval"), env)
        if value is not None:
            print(repr(value))

def rlm(context, question, max_turns=8):
    env = {"context": context, "llm_query": llm_query, "re": re}
    messages = [{"role": "system", "content": SYSTEM.format(n=len(context))}, {"role": "user", "content": question}]
    LOG.append(messages)                                  # every turn is kept for inspection
    for _ in range(max_turns):
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})
        m = re.search(r"FINAL(_VAR)?\((.*?)\)\s*$", reply.strip(), re.S)
        if m:
            return str(env.get(m.group(2).strip(), "")) if m.group(1) else m.group(2).strip()
        code = "\n".join(re.findall(r"```(?:python|repl)\n(.*?)```", reply, re.S))
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                run(code, env)
            out = buf.getvalue() or "(no output)"
        except Exception as e:
            out = f"error: {type(e).__name__}: {e}"
        messages.append({"role": "user", "content": "Output:\n" + out[:2000]})
    return None

def direct(context, question, budget=8000):
    ids = tok(context).input_ids[:budget]
    return chat([{"role": "user", "content": tok.decode(ids) + "\n\n" + question}], max_new=60)

def scripted(context, kind, term=None):
    """The same sub-calls, but the decomposition is a fixed program, not written by the model."""
    if kind == "needle":                                  # map over 19,000-character chunks, keep any reply
        hits = [llm_query(context[i:i + 19000] + "\n\nDoes this text mention an archive access code? "
                          "If so, reply with the code only; otherwise reply NONE.", max_new=20)
                for i in range(0, len(context), 19000)]
        return [h.strip() for h in hits if "NONE" not in h.upper()]
    yes = 0                                               # count: sub-calls per post (in pieces), then add up
    for post in context.split("=== POST:")[1:]:
        yes += any(llm_query(post[i:i + 19000] + f"\n\nDoes this text mention '{term}' (case-insensitive)? "
                             "Reply YES or NO.", max_new=5).strip().upper().startswith("YES")
                   for i in range(0, len(post), 19000))
    return yes

# Tasks with answers a program can check.
tasks = []
for k in range(5):                                        # a planted fact somewhere in the corpus
    code = f"{rng.choice(['amber', 'cobalt', 'violet', 'ochre'])}-{rng.randint(100, 999)}"
    corpus = posts[:]
    corpus[rng.randrange(len(posts))] += f"\nNote for readers: the archive access code is {code}.\n"
    tasks.append(("needle", "\n".join(corpus), "What is the archive access code mentioned in the text?", code, None))
for term in ("KV cache", "GPU", "tokenizer", "reward", "attention"):   # count across the whole corpus
    n = sum(term.lower() in p.lower() for p in posts)
    tasks.append(("count", "\n".join(posts), f"How many of the posts (each starts with a line '=== POST: ... ===') "
                  f"mention '{term}' (case-insensitive)? Answer with a number.", str(n), term))
print(f"corpus: {len(posts)} posts, {len(tasks[5][1]):,} characters (~{len(tok(tasks[5][1]).input_ids):,} tokens)\n")
print(f"{'task':7s} {'truth':>10s} {'direct':>16s} {'RLM':>16s} {'calls':>6s} {'sub':>5s} {'fixed program':>24s} "
      f"{'calls':>6s}  minutes: direct, RLM, fixed", flush=True)
score, subs = {"direct": 0, "rlm": 0, "fixed": 0}, []
for kind, corpus, q, truth, term in tasks:
    mins = []
    chat.calls = 0; t = time.time(); d = direct(corpus, q); mins.append((time.time() - t) / 60)
    chat.calls = 0; SUB.clear(); t = time.time(); r = rlm(corpus, q) or ""; mins.append((time.time() - t) / 60)
    rlm_calls, rlm_sub = chat.calls, len(SUB); subs.append({"task": q, "rlm": SUB[:]})
    chat.calls = 0; SUB.clear(); t = time.time(); f = scripted(corpus, kind, term); mins.append((time.time() - t) / 60)
    if kind == "needle":                                  # right only if exactly one distinct code comes back, the true one
        ok_d, ok_r = truth in d, truth in r
        codes = {c for h in f for c in re.findall(r"[a-z]+-\d{3}", h)}
        ok_f = codes == {truth}
        f_show = f"{len(f)} hits, {len(codes)} codes"
    else:
        ok_d, ok_r = re.findall(r"\d+", d)[:1] == [truth], re.sub(r"\D", "", r) == truth
        ok_f = str(f) == truth
        f_show = str(f)
    score["direct"] += ok_d; score["rlm"] += ok_r; score["fixed"] += ok_f
    show = lambda ok, s: "right" if ok else "wrong: " + " ".join(s.split())[:9]
    print(f"{kind:7s} {truth:>10s} {show(ok_d, d):>16s} {show(ok_r, r):>16s} {rlm_calls:>6d} {rlm_sub:>5d} "
          f"{('right' if ok_f else 'wrong') + ': ' + f_show:>24s} {chat.calls:>6d}  "
          f"{', '.join(f'{m:.1f}' for m in mins)}", flush=True)
    subs[-1]["fixed"] = SUB[:]
    json.dump({"rlm": LOG, "sub_calls": subs}, open("rlm_transcripts.json", "w"))   # saved after every task
print(f"\nright: direct {score['direct']} of {len(tasks)}, RLM {score['rlm']} of {len(tasks)}, "
      f"fixed program {score['fixed']} of {len(tasks)}")