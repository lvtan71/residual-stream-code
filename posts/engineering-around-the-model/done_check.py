import json
import random
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# Two harnesses around the same model (Qwen2.5-1.5B-Instruct) and tool (a
# calculator), on the 20 warehouse problems from "Evaluating and Observing Agents", 3 runs each.
#   plain: stop at the first reply that doesn't call a tool (from "Evaluating and Observing Agents").
#   checked: a definition of done the harness can verify without knowing
#   the answer: the reply's final number must equal the calculator's most
#   recent result. If not, the harness says so and the agent continues; if
#   the budget runs out, it returns the last answer anyway (unverified).
# Up to 8 model calls per run; the step at which each run finished is kept,
# so success under any smaller budget can be read off afterwards.
PATH = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                         revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
tok = AutoTokenizer.from_pretrained(PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16).to(DEVICE).eval()
TOOLS = [{"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic "
          "expression with + - * / and parentheses, and return the number.",
          "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                         "required": ["expression"]}}}]
def calculator(expression):
    if not re.fullmatch(r"[\d\s+\-*/().]+", expression or ""):
        return "error: only numbers and + - * / ( ) are allowed"
    return str(round(eval(expression), 6))

rng = random.Random(0)                                  # the same 20 problems as "Evaluating and Observing Agents"
tasks = []
for _ in range(20):
    a, b, c, d = rng.randint(1000, 9999), rng.randint(100, 999), rng.randint(3, 9), rng.randint(100, 999)
    tasks.append((f"A warehouse holds {a} boxes. It ships out {b} boxes, then receives {c} deliveries of "
                  f"{d} boxes each. How many boxes does it hold now?", a - b + c * d))

def last_number(text):
    nums = re.findall(r"-?\d[\d,]*", text)
    return int(nums[-1].replace(",", "")) if nums else None

@torch.no_grad()
def run(question, seed, checked, max_calls=8):
    torch.manual_seed(seed)
    messages = [{"role": "system", "content": "Use the calculator tool for every arithmetic step. "
                 "End with: Answer: <number>"}, {"role": "user", "content": question}]
    last, calls, answer, given = None, 0, None, []    # given: (call number, answer) for every answer the agent gives
    while calls < max_calls:
        prompt = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        out = model.generate(ids, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=1.0, top_k=0,
                             repetition_penalty=1.1)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        calls += 1
        messages.append({"role": "assistant", "content": text})
        tool_calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
        if tool_calls:
            for c in tool_calls:
                try:
                    r = calculator(json.loads(c)["arguments"]["expression"])
                except (ValueError, KeyError, TypeError):
                    r = "error: malformed tool call"
                if not r.startswith("error"):
                    last = float(r)
                messages.append({"role": "tool", "content": r})
            continue
        answer = last_number(text)
        given.append((calls, answer))
        if not checked or (answer is not None and last is not None and answer == last):
            return answer, calls, messages, given
        messages.append({"role": "user", "content": f"Your answer {answer} is not the calculator's last result. "
                         "Use the calculator for each step, then give its final result as the answer."})
    return answer, calls, messages, given                # budget exhausted: the last answer, unverified

log = {}
for name in ("plain", "checked"):
    rows = []
    for i, (q, truth) in enumerate(tasks):
        for k in range(3):
            ans, calls, msgs, given = run(q, seed=100 * i + k, checked=name == "checked")
            rows.append({"task": i, "right": ans == truth, "calls": calls, "answer": ans, "truth": truth, "given": given,
                         "transcript": [m["content"][:400] for m in msgs[1:]]})
    log[name] = rows
    print(f"{name:8s} right {np.mean([r['right'] for r in rows]):.0%}", flush=True)
json.dump(log, open("runs.json", "w"))

print(f"\n{'harness':8s} " + " ".join(f"{'budget ' + str(b):>10s}" for b in (2, 3, 4, 6, 8)) + "   (share right)")
for name, rows in log.items():
    def at_budget(r, b):
        # Cut at b calls, the harness returns the run's answer if it finished
        # by then, else the last answer the agent had given (unverified).
        if r["calls"] <= b:
            return r["answer"]
        earlier = [a for c, a in r["given"] if c <= b]
        return earlier[-1] if earlier else None
    print(f"{name:8s} " + " ".join(f"{np.mean([at_budget(r, b) == r['truth'] for r in rows]):>10.0%}"
                                   for b in (2, 3, 4, 6, 8)))
for name, rows in log.items():
    calls = [r["calls"] for r in rows]
    print(f"{name}: model calls per run, mean {np.mean(calls):.1f}, max {max(calls)}; "
          f"used all 8 calls {sum(c == 8 for c in calls)} of {len(rows)}")
plain, checked = log["plain"], log["checked"]
print(f"paired: the check fixed {sum(not p['right'] and c['right'] for p, c in zip(plain, checked))} plain failures "
      f"and broke {sum(p['right'] and not c['right'] for p, c in zip(plain, checked))} plain successes")