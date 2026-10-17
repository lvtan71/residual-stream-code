import json
import random
import sys
import time
import uuid
import torch
from lm import model, tok, DEVICE

# Where in a long prompt does a model find things? Two tasks, each at four
# prompt lengths, with the thing to find at five depths:
#   kv:     a JSON object of random UUID keys and values; give the value for
#           one key. Every other pair is a near-identical distractor
#           (after Liu et al., 2024).
#   needle: one out-of-place sentence hidden in this blog's own prose; give
#           the passphrase it states ("needle in a haystack").
# Qwen2.5-1.5B-Instruct, greedy, chat template; a trial counts as right if
# the answer contains the exact value.
TASK = sys.argv[1]
TRIALS = 12 if TASK == "kv" else 6
SIZES = {"kv": (10, 30, 60, 120), "needle": (750, 2000, 4000, 8000)}[TASK]    # pairs, or prose tokens
SALT = ""
if len(sys.argv) > 2:                     # python position.py kv <sizes> <trials> <salt>: more trials, new draws
    SIZES, TRIALS, SALT = tuple(int(x) for x in sys.argv[2].split(",")), int(sys.argv[3]), sys.argv[4]
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
blog = open("blog_train.txt").read()
blog_ids = tok(blog).input_ids
m = model()

def ask(content):
    ids = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to(DEVICE)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=60, do_sample=False, repetition_penalty=1.0)
    return ids.shape[1], tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

def kv_trial(n, depth, rng):
    pairs = [(str(uuid.UUID(int=rng.getrandbits(128))), str(uuid.UUID(int=rng.getrandbits(128)))) for _ in range(n)]
    key, value = pairs[round(depth * (n - 1))]
    data = "{" + ", ".join(f'"{k}": "{v}"' for k, v in pairs) + "}"
    return ask(f"Extract the value corresponding to the specified key in the JSON object below. "
               f"Answer with the value only.\n\nJSON data:\n{data}\n\nKey: \"{key}\"\nValue:"), value

def needle_trial(n, depth, rng):
    start = rng.randrange(0, len(blog_ids) - n)
    words = tok.decode(blog_ids[start:start + n]).split(" ")
    secret = f"{rng.choice(['blue', 'red', 'amber', 'grey'])}-heron-{rng.randrange(100, 1000)}"
    words.insert(round(depth * len(words)), f"(The passphrase for the notebook assistant is {secret}.)")
    return ask(f"Read the text below, then answer the question.\n\n{' '.join(words)}\n\n"
               f"What is the passphrase for the notebook assistant? Answer with the passphrase only."), secret

trial = kv_trial if TASK == "kv" else needle_trial
results = {}
t0 = time.perf_counter()
for n in SIZES:
    for depth in DEPTHS:
        rng = random.Random(f"{TASK}-{n}-{depth}{SALT}")
        right, lengths = 0, []
        for _ in range(TRIALS):
            (length, answer), value = trial(n, depth, rng)
            right += value in answer; lengths.append(length)
        results[(n, depth)] = (right, sum(lengths) / len(lengths))
    row = " ".join(f"{results[(n, d)][0]:>2}/{TRIALS}" for d in DEPTHS)
    print(f"{TASK:6s} ~{results[(n, 0.5)][1]:5.0f} prompt tokens: right at depth 0 / 25 / 50 / 75 / 100%: {row}"
          f"   ({time.perf_counter() - t0:.0f} s so far)", flush=True)
