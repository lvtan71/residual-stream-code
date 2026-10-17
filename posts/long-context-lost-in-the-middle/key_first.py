import random
import uuid
import torch
from lm import model, tok, DEVICE

# The 12 key-value objects of position.py's 8,100-token, 50%-depth cell
# (the first run's random draws, regenerated from the same seed), asked two
# ways: as in position.py, and with the key also stated before the data.
# Every answer is classified: right, another pair's value, the right value
# with a character or two wrong, or none of these.
m = model()

def ask(content):
    ids = tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to(DEVICE)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=60, do_sample=False, repetition_penalty=1.0)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

def close(a, b):                              # equal length and at most 2 characters differ
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= 2

n, depth = 120, 0.5
rng = random.Random(f"kv-{n}-{depth}")
target = round(depth * (n - 1))
counts = {"after only": {}, "before and after": {}}
for trial in range(12):
    pairs = [(str(uuid.UUID(int=rng.getrandbits(128))), str(uuid.UUID(int=rng.getrandbits(128)))) for _ in range(n)]
    key, value = pairs[target]
    data = "{" + ", ".join(f'"{k}": "{v}"' for k, v in pairs) + "}"
    task = "Extract the value corresponding to the specified key in the JSON object below. Answer with the value only."
    for way, content in (("after only", f"{task}\n\nJSON data:\n{data}\n\nKey: \"{key}\"\nValue:"),
                         ("before and after", f"{task}\n\nKey: \"{key}\"\n\nJSON data:\n{data}\n\nKey: \"{key}\"\nValue:")):
        answer = ask(content)
        values = [v for _, v in pairs]
        if value in answer:
            kind = "right"
        elif any(v in answer for v in values):
            kind = f"another pair's value (pair {next(i for i, v in enumerate(values) if v in answer)}, target {target})"
        elif any(close(w.strip('"`.'), value) for w in answer.split()):
            kind = "right value, a character or two wrong"
        else:
            kind = f"none of these: {answer[:40]!r}"
        counts[way][kind] = counts[way].get(kind, 0) + 1
for way, c in counts.items():
    print(f"{way}: " + "; ".join(f"{k}: {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])))
