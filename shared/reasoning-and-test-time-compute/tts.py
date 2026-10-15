import re
import sys
import time
from collections import Counter
import numpy as np
import torch
from lm import model, tok, DEVICE

# Test-time compute on a problem set whose answers can be checked: 100
# three-step arithmetic word problems generated from templates, each with a
# known integer answer. Qwen2.5-1.5B-Instruct answers each one
#   direct:  "answer with just the number", greedy;
#   chain:   "think step by step, then 'Answer: <number>'", greedy;
#   sampled: the same chain prompt, 16 samples at temperature 0.7, scored
#            one at a time and by majority vote over the first k.
# Every decoding parameter is passed explicitly ("Decoding").
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(1234)                                   # the problems are fixed across seeds
TEMPLATES = [
    ("A library has {a} shelves with {b} books on each. It lends out {c} books and receives {d} new ones. How many books does it have now?", lambda a, b, c, d: a * b - c + d),
    ("Minh buys {a} packs of {b} stickers, gives {c} stickers to each of his {d} friends, and keeps the rest. How many does he keep?", lambda a, b, c, d: a * b - c * d),
    ("A train has {a} cars with {b} seats each. {c} seats are broken, and the remaining seats are shared equally by {d} tour groups. How many seats does each group get?", None),
    ("A farm collects {a} eggs a day for {b} days, then sells them in cartons of {d}. {c} eggs break first. How many full cartons can it sell?", lambda a, b, c, d: (a * b - c) // d),
]
problems = []
while len(problems) < 100:
    k = len(problems) % len(TEMPLATES)
    text, f = TEMPLATES[k]
    a, b, c, d = (int(x) for x in rng.integers([3, 4, 2, 2], [15, 25, 30, 9]))
    if k == 2:                                                      # make the division exact
        if (a * b - c) % d or a * b <= c:
            continue
        f = lambda a, b, c, d: (a * b - c) // d
    answer = f(a, b, c, d)
    if answer <= 0:
        continue
    problems.append((text.format(a=a, b=b, c=c, d=d), answer))

m = model()
torch.manual_seed(seed)
def chat(content):
    return tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True)["input_ids"].to(DEVICE)
def number(text):
    after = text.split("Answer:")[-1]
    nums = re.findall(r"-?\d[\d,]*", after)
    return int(nums[-1].replace(",", "")) if nums else None

GEN = dict(do_sample=False, repetition_penalty=1.0, top_k=None, top_p=None, temperature=None, max_new_tokens=320)
SAMPLE = dict(do_sample=True, temperature=0.7, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=320,
              num_return_sequences=16)
res = {"direct": [], "chain": [], "samples": []}
tokens = {"direct": [], "chain": [], "samples": []}
t0 = time.perf_counter()
for q, ans in problems:
    if seed == 0:                                                   # greedy answers don't depend on the seed
        for kind, prompt in (("direct", q + " Answer with just the number."),
                             ("chain", q + " Think step by step, then give the final answer as 'Answer: <number>'.")):
            x = chat(prompt)
            out = m.generate(x, **GEN)[0, x.shape[1]:]
            res[kind].append(number(tok.decode(out, skip_special_tokens=True)) == ans)
            tokens[kind].append(int((out != tok.pad_token_id).sum()))
    x = chat(q + " Think step by step, then give the final answer as 'Answer: <number>'.")
    outs = m.generate(x, **SAMPLE)[:, x.shape[1]:]
    res["samples"].append([number(tok.decode(o, skip_special_tokens=True)) for o in outs])
    tokens["samples"].append([int((o != tok.pad_token_id).sum()) for o in outs])
print(f"seed {seed}: {len(problems)} problems, {time.perf_counter() - t0:.0f} s")
if seed == 0:
    for kind in ("direct", "chain"):
        print(f"  greedy, {kind:6s}: {np.mean(res[kind]):.0%} right, {np.mean(tokens[kind]):.0f} tokens per answer")
S, truth = res["samples"], [a for _, a in problems]
print(f"  sampled chain, one sample: {np.mean([[s == t for s in ss] for ss, t in zip(S, truth)]):.0%} right, "
      f"{np.mean(tokens['samples']):.0f} tokens per answer")
for k in (1, 2, 4, 8, 16):
    vote = [Counter([s for s in ss[:k] if s is not None]).most_common(1) for ss in S]
    acc = np.mean([bool(v) and v[0][0] == t for v, t in zip(vote, truth)])
    any_right = np.mean([t in ss[:k] for ss, t in zip(S, truth)])
    print(f"  majority of {k:2d}: {acc:.0%} right ({k * np.mean(tokens['samples']):5.0f} tokens per question); "
          f"some sample right: {any_right:.0%}")
