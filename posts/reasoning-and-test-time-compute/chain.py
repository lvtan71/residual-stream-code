import json
import os
import re
import numpy as np
import torch
from lm import model, tok, DEVICE

# Serial steps, with Qwen2.5-1.5B-Instruct. Each problem is a start number
# and k operations to apply in order ("add 17", "multiply by 3", ...), with
# a known answer; 40 problems for each k in 1, 2, 3, 4, 6, 8. Two choices,
# crossed: one call for the whole problem or one call per operation (a
# chain), and answering with just the number or working it out first.
#   direct:          one call, "answer with just the number";
#   reasoned:        one call, "work it out step by step, then 'Answer: <n>'";
#   chained:         a call per operation, each given only the previous
#                    result, answering with just the number;
#   chained, reasoned: the same chain, each call working its step out first.
# Then two ways to check a chained step before the next one trusts it:
#   agreement:       sample the step twice (temperature 0.7) and accept a
#                    value only if both agree, up to 3 pairs; a check that
#                    doesn't know the answer;
#   oracle:          compare with the true value and retry (sampled) up to 3
#                    times; the best case, since code that knows the answer
#                    wouldn't need the model.
# Greedy calls pass every decoding parameter explicitly ("Decoding"). The two
# checked arms sample, so they run with three seeds.
N = int(os.environ.get("N", 40))
rng = np.random.default_rng(7)                                       # the same problems as before
OPS = [("add {v}", lambda x, v: x + v, (3, 60)), ("subtract {v}", lambda x, v: x - v, (3, 40)),
       ("multiply by {v}", lambda x, v: x * v, (2, 6))]
def make(k):
    x = int(rng.integers(5, 50)); start, steps = x, []
    for _ in range(k):
        name, f, (lo, hi) = OPS[int(rng.integers(0, 3))]
        v = int(rng.integers(lo, hi)); steps.append((name.format(v=v), f, v)); x = f(x, v)
    return start, steps, x
KS = (1, 2, 3, 4, 6, 8)
problems = {k: [make(k) for _ in range(N)] for k in KS}

m = model()
GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
SAMPLE = dict(do_sample=True, temperature=0.7, top_p=1.0, top_k=0, repetition_penalty=1.0)
LOG = {"truncated": 0, "calls": 0}
def ask(text, tokens, sample=False):
    x = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                return_tensors="pt", return_dict=True)["input_ids"].to(DEVICE)
    out = m.generate(x, max_new_tokens=tokens, **(SAMPLE if sample else GREEDY))
    LOG["calls"] += 1; LOG["truncated"] += out.shape[1] - x.shape[1] == tokens   # hit the token cap
    return tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)
def number(text, need_answer=False):
    if need_answer and "Answer:" not in text:
        return None                                                  # a reasoned reply must end with its answer
    nums = re.findall(r"-?\d+", text.split("Answer:")[-1].replace(",", ""))
    return int(nums[-1]) if nums else None
def describe(start, steps):
    return f"Start with {start}. " + " ".join(f"Then {s}." for s, _, _ in steps)
STEP = "Take {x} and {s}. Answer with just the number."
STEP_R = "Take {x} and {s}. Work it out, then give the result as 'Answer: <number>'."

def chain(start, steps, step_fn, record=None):
    x = start
    for s, f, v in steps:
        y = step_fn(x, s, f(x, v))
        if record is not None:
            record.append({"x": x, "op": s, "right": y == f(x, v)})
        x = y if y is not None else x
    return x

def agree(x, s, truth):
    for _ in range(3):                                               # accept a value two samples agree on
        a, b = (number(ask(STEP.format(x=x, s=s), 12, sample=True)) for _ in range(2))
        if a is not None and a == b:
            return a
    return a                                                         # no agreement: keep the last sample

def oracle(x, s, truth):
    for attempt in range(4):
        y = number(ask(STEP.format(x=x, s=s), 12, sample=attempt > 0))
        if y == truth:
            return y
    return y

results, steps_log = {}, []
for k in KS:
    r = {}
    for name in ("direct", "reasoned", "chained", "chained, reasoned"):
        LOG.update(truncated=0, calls=0); right = 0
        for start, steps, answer in problems[k]:
            if name == "direct":
                y = number(ask(describe(start, steps) + " What is the result? Answer with just the number.", 12))
            elif name == "reasoned":
                y = number(ask(describe(start, steps) + " Work it out step by step, then give the final "
                               "result as 'Answer: <number>'.", 60 + 40 * k), need_answer=True)
            elif name == "chained":
                y = chain(start, steps, lambda x, s, t: number(ask(STEP.format(x=x, s=s), 12)), steps_log)
            else:
                y = chain(start, steps, lambda x, s, t: number(ask(STEP_R.format(x=x, s=s), 120), need_answer=True))
            right += y == answer
        r[name] = (right, LOG["calls"] / N, LOG["truncated"])
    for name, fn in (("checked by agreement", agree), ("checked by oracle", oracle)):
        rights, calls = [], []
        for seed in (0, 1, 2):
            torch.manual_seed(seed); LOG.update(calls=0)
            rights.append(sum(chain(st, sp, fn) == ans for st, sp, ans in problems[k])); calls.append(LOG["calls"] / N)
        r[name] = (rights, np.mean(calls), None)
    results[k] = r
    print(f"k = {k}", flush=True)
    for name, (right, calls, trunc) in r.items():
        shown = f"{right / N:4.0%}" if isinstance(right, int) else "/".join(f"{x / N:.0%}" for x in right) + " (3 seeds)"
        print(f"   {name:20s} {shown:>22s}   {calls:5.1f} calls per problem" + (f", {trunc} hit the token cap" if trunc else ""),
              flush=True)
json.dump(steps_log, open("chain_steps.json", "w"))                   # every greedy chained step, for analysis
mult = [s for s in steps_log if s["op"].startswith("multiply")]
other = [s for s in steps_log if not s["op"].startswith("multiply")]
big = [s for s in steps_log if abs(s["x"]) >= 1000]
print(f"\none greedy chained step right: {np.mean([s['right'] for s in steps_log]):.1%} overall; "
      f"multiply {np.mean([s['right'] for s in mult]):.1%}, add or subtract {np.mean([s['right'] for s in other]):.1%}; "
      f"starting from a number of 1,000 or more {np.mean([s['right'] for s in big]):.1%} ({len(big)} steps)")
