import json
import random
import re
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# Three agent architectures on the same tasks, model and tools.
# The world: 12 fictional companies with founding years, headcounts and
# cities, so nothing can be answered from memory. Tools: lookup(company,
# field) and calculator(expression). Each question needs 2-3 lookups and
# some arithmetic. Model: Qwen2.5-3B-Instruct, temperature 0.7, 3 runs each.
#   ReAct: think and act in one loop until it answers.
#   Plan-and-execute: write a numbered plan first (no tools), then carry out
#     each step in its own short tool loop, passing on each call made and
#     its result, verbatim; then a final tool loop answers from them.
#   Reflection: ReAct, then a critique of the transcript; if the critic
#     finds a problem, one more ReAct attempt with the critique attached.
# A reply counts as an answer only if it contains "Answer: <number>".
PATH = snapshot_download("Qwen/Qwen2.5-3B-Instruct",
                         revision="aa8e72537993ba99e69dfaafa59ed015b17504d1")
tok = AutoTokenizer.from_pretrained(PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16).to(DEVICE).eval()

rng = random.Random(1)
names = ["Arvelo", "Brindle", "Cobaltic", "Dunmore", "Elstree", "Farrow", "Gilden", "Harrowgate", "Istria",
         "Juniper Labs", "Kestrel", "Lumen Forge"]
cities = ["Lyon", "Osaka", "Porto", "Tartu", "Quito", "Perth"]
db = {n: {"founded": rng.randint(1950, 2020), "employees": rng.randint(40, 9000), "city": rng.choice(cities)}
      for n in names}
TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up one field of a company: "
          "the year it was founded, its number of employees, or its city.", "parameters": {"type": "object", "properties": {
          "company": {"type": "string"}, "field": {"type": "string", "enum": ["founded", "employees", "city"]}},
          "required": ["company", "field"]}}},
         {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression.",
          "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]
def tool(name, args):
    if name == "lookup":                      # errors say exactly what would have worked
        company, field = str(args.get("company", "")).strip(), args.get("field")
        if company not in db:
            return f"error: no company named {company!r}; companies are: {', '.join(names)}"
        if field not in db[company]:
            return f"error: no field {field!r}; fields are: founded, employees, city"
        return str(db[company][field])
    if name == "calculator":
        e = str(args.get("expression", ""))
        return str(round(eval(e), 4)) if re.fullmatch(r"[\d\s+\-*/().]+", e) else "error: arithmetic only"
    return "error: unknown tool"

tasks = []
for _ in range(20):
    a, b, c = rng.sample(names, 3)
    kind = rng.randrange(3)
    if kind == 0:
        tasks.append((f"How many years older is {a} than {b}? (Use founding years; negative if younger.)",
                      db[b]["founded"] - db[a]["founded"]))
    elif kind == 1:
        tasks.append((f"How many employees do {a}, {b} and {c} have in total?",
                      db[a]["employees"] + db[b]["employees"] + db[c]["employees"]))
    else:
        tasks.append((f"How many more employees does {a} have than {b}? (Negative if fewer.)",
                      db[a]["employees"] - db[b]["employees"]))

SYS = "You answer questions using the tools. Never guess a value; look it up. End with: Answer: <number>"
@torch.no_grad()
def gen(messages, tools=TOOLS, max_new=384):
    ids = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)["input_ids"].to(DEVICE)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=1.0, top_k=0)
    gen.tokens += out.shape[1]                # tokens processed: prompt + output, every call
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

def tool_loop(messages, max_calls=8):
    """Run model calls, executing tool calls, until a reply without one."""
    tool_loop.calls = []                      # each executed call with its result, for plan-and-execute
    for _ in range(max_calls):
        text = gen(messages)
        messages.append({"role": "assistant", "content": text})
        calls = re.findall(r"<tool_call>\s*(\{.*?)\s*</tool_call>", text, re.S)
        if not calls:
            return text
        for c in calls:
            try:                                  # tolerate trailing junk such as an extra "}"
                call = json.JSONDecoder().raw_decode(c)[0]; r = tool(call["name"], call.get("arguments", {}))
                tool_loop.calls.append(f"{call['name']}({json.dumps(call.get('arguments', {}))}) = {r}")
            except (ValueError, KeyError, TypeError, AttributeError):
                r = ('error: could not parse the tool call; write exactly <tool_call>{"name": "lookup", '
                     '"arguments": {"company": "...", "field": "employees"}}</tool_call>')
            messages.append({"role": "tool", "content": r})
    return ""

def react(q):
    return tool_loop([{"role": "system", "content": SYS}, {"role": "user", "content": q}])

def plan_execute(q):
    plan = gen([{"role": "user", "content": "Write a short numbered plan (one tool action per line, no answers) "
                 f"for answering this with lookup and calculator tools:\n{q}"}], tools=None, max_new=200)
    steps = [l for l in plan.splitlines() if re.match(r"\s*(\d+[.):]|[-*]|step\s*\d+)", l, re.I)][:6]
    steps = steps or [l for l in plan.splitlines() if l.strip()][:6]
    LOG["plan"] = steps
    notes = []                                # each step's tool calls and results, verbatim
    for s in steps:
        msgs = [{"role": "system", "content": "Do exactly this one step with the tools."},
                {"role": "user", "content": f"Question: {q}\nResults so far: {notes}\nStep: {s}"}]
        tool_loop(msgs, max_calls=3)
        notes += tool_loop.calls
    LOG["notes"] = notes
    return tool_loop([{"role": "system", "content": SYS}, {"role": "user", "content":
                      f"{q}\nTool results from the plan's steps: {notes}"}])

def reflect(q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    first = tool_loop(msgs)
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in msgs[1:])[-3000:]
    critique = gen([{"role": "user", "content": "Check this attempt. Were all values looked up (not guessed) and is the "
                     f"arithmetic right? Reply OK, or describe the error.\n\n{transcript}"}], tools=None, max_new=150)
    LOG["critique"] = critique
    if re.match(r"\W*OK\b", critique.strip().upper()):
        return first
    return tool_loop([{"role": "system", "content": SYS}, {"role": "user", "content":
                       f"{q}\nA previous attempt was checked and this problem was found: {critique}"}])

def answer(text):
    m = re.search(r"Answer:\s*(-?[\d,]+)", text or "")
    return int(m.group(1).replace(",", "")) if m else None

print(f"{len(tasks)} questions x 3 runs; Qwen2.5-3B-Instruct\n")
print(f"{'architecture':18s} {'right':>6s} {'tokens processed/question':>26s} {'seconds/question':>17s}")
import os
LOG = {}
done = {}                                     # each trial is logged, so a stopped run resumes
if os.path.exists("arch_trials_v3.jsonl"):
    for l in open("arch_trials_v3.jsonl"):
        t = json.loads(l); done[(t["arch"], t["i"], t["k"])] = t
for name, f in (("ReAct", react), ("plan-and-execute", plan_execute), ("reflection", reflect)):
    right, toks, secs = [], [], []
    for i, (q, truth) in enumerate(tasks):
        for k in range(3):
            if (name, i, k) not in done:
                torch.manual_seed(1000 * i + k); gen.tokens = 0; t0 = time.perf_counter(); LOG.clear()
                reply = f(q)
                t = {"arch": name, "i": i, "k": k, "pred": answer(reply), "truth": truth, "tokens": gen.tokens,
                     "secs": time.perf_counter() - t0, "reply": reply[-600:], **LOG}
                with open("arch_trials_v3.jsonl", "a") as fh:
                    fh.write(json.dumps(t) + "\n")
                done[(name, i, k)] = t
            t = done[(name, i, k)]
            right.append(t["pred"] == truth); toks.append(t["tokens"]); secs.append(t["secs"])
    print(f"{name:18s} {np.mean(right):6.0%} {np.mean(toks):26,.0f} {np.mean(secs):17.1f}", flush=True)
