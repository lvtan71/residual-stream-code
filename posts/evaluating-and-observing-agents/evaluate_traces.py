import json
import re
import numpy as np
from math import comb

# Evaluating the agent from its traces: every fact below is read from the
# span records (message contents, finish reasons, tool calls); the only
# other input is the answer key.
results = json.load(open("traces.json"))
key = {r["task"]: r["truth"] for r in results}

def facts(r):
    spans = r["trace"]["spans"]
    chats = [s["attributes"] for s in spans if s["attributes"]["gen_ai.operation.name"] == "chat"]
    tools = [s["attributes"] for s in spans if s["attributes"]["gen_ai.operation.name"] == "execute_tool"]
    final = chats[-1]["gen_ai.output.messages"][0]["content"]
    strict = re.search(r"Answer:\s*\$?(-?[\d,]+)", final)
    nums = re.findall(r"-?\d[\d,]*", final)
    return {"final": final, "tools": tools,
            "truncated": any(c["gen_ai.response.finish_reasons"] == ["length"] for c in chats),
            "strict": int(strict.group(1).replace(",", "")) if strict else None,
            "lenient": int(nums[-1].replace(",", "")) if nums else None,
            "tool_has_truth": str(key[r["task"]]) in [t.get("gen_ai.tool.call.result", "").split(".")[0] for t in tools]}
F = [(r, facts(r)) for r in results]

pass_at = lambda n, c, k: 1 - comb(n - c, k) / comb(n, k)
pass_hat = lambda n, c, k: comb(c, k) / comb(n, k)
for name, field in (("as asked ('Answer: <number>')", "strict"), ("by the last number in the reply", "lenient")):
    per = {}
    for r, f in F:
        per.setdefault(r["task"], []).append(f[field] == key[r["task"]])
    print(f"outcome scored {name}, 20 tasks x 5 runs:")
    for k in (1, 3, 5):
        print(f"  pass@{k} {np.mean([pass_at(5, sum(v), k) for v in per.values()]):.2f}   "
              f"pass^{k} {np.mean([pass_hat(5, sum(v), k) for v in per.values()]):.2f}")
    print(f"  tasks solved in all 5 runs: {sum(sum(v) == 5 for v in per.values())} of 20")

def trajectory(f, truth):
    right = f["lenient"] == truth
    if f["truncated"]:
        return "cut off by the token budget" + (" (still right)" if right else "")
    if right:
        return "right, the calculator produced the answer" if f["tool_has_truth"] else "right, the calculator didn't produce it"
    if f["tools"] and not f["tool_has_truth"]:
        return "wrong: asked the calculator the wrong question"
    if f["tool_has_truth"]:
        return "wrong: calculator had it, reply differs"
    return "wrong: no calculator"
cats = {}
for r, f in F:
    c = trajectory(f, key[r["task"]]); cats[c] = cats.get(c, 0) + 1
print("\ntrajectories (100 runs):")
for c, v in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {v:3d}  {c}")
calls = [len(f["tools"]) for _, f in F]
print(f"  calculator calls per run: none {calls.count(0)}, one {calls.count(1)}, two or more {sum(c >= 2 for c in calls)}"
      f" (the instructions asked for one per arithmetic step: two)")
print(f"  runs with no 'Answer:' line: {sum(f['strict'] is None for _, f in F)}; of those, replies saying "
      f"'the answer is': {sum(f['strict'] is None and 'answer is' in f['final'].lower() for _, f in F)}")

spans = [s for r in results for s in r["trace"]["spans"]]
by = lambda op: [s for s in spans if s["attributes"]["gen_ai.operation.name"] == op]
dur = lambda ss: np.array([s["end"] - s["start"] for s in ss])
chat, tool, agents = by("chat"), by("execute_tool"), by("invoke_agent")
tin = sum(s["attributes"]["gen_ai.usage.input_tokens"] for s in chat)
tout = sum(s["attributes"]["gen_ai.usage.output_tokens"] for s in chat)
right = sum(f["lenient"] == key[r["task"]] for r, f in F)
print(f"\ndashboard: {len(agents)} runs, {len(chat)} model calls, {len(tool)} tool calls")
print(f"  per run: {len(chat) / len(agents):.1f} model calls, {tin / len(agents):,.0f} tokens in, {tout / len(agents):.0f} out; "
      f"per successful run: {(tin + tout) / right:,.0f} tokens")
print(f"  run latency p50 {np.median(dur(agents)):.1f} s, p95 {np.percentile(dur(agents), 95):.1f} s; "
      f"model call p50 {np.median(dur(chat)):.2f} s; tool call p50 {np.median(dur(tool)) * 1e6:.0f} us")

r, f = next((r, f) for r, f in F if trajectory(f, key[r["task"]]).startswith("wrong: asked"))
print(f"\none trace (task {r['task']}, right answer {key[r['task']]}):")
t0 = r["trace"]["spans"][0]["start"]
for s in r["trace"]["spans"]:
    a = s["attributes"]
    extra = (f"in {a['gen_ai.usage.input_tokens']} out {a['gen_ai.usage.output_tokens']} tokens, "
             f"finish {a['gen_ai.response.finish_reasons'][0]}" if a["gen_ai.operation.name"] == "chat" else
             f"{a.get('gen_ai.tool.call.arguments')} -> {a.get('gen_ai.tool.call.result')}"
             if a["gen_ai.operation.name"] == "execute_tool" else "")
    print(f"{'  ' if s['parent_id'] is None else '      '}{s['name']:38s} {(s['start'] - t0) * 1e3:7.0f} ms  "
          f"+{(s['end'] - s['start']) * 1e3:6.0f} ms  {extra}")
print(f"  final reply: {f['final']!r}")