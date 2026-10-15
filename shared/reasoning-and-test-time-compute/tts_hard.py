import os
import sys
from collections import Counter
import numpy as np
import torch
from lm import model, tok, DEVICE
sys.path.insert(0, "../post33")

# The same measurement on harder problems, where one chain isn't enough:
# "Chain-of-Thought and Prompt Chaining"'s serial arithmetic with eight dependent operations (40
# problems), on which one greedy chain is right 92% of the time. 16
# chains sampled at temperature 0.7 per problem; the majority answer of
# the first k, and whether any of them is right (what a perfect verifier
# could pick, best-of-N).
src = open("chain.py").read()
exec(src[src.index("N = int("):src.index("m = model()")])          # the problem generator of "Chain-of-Thought and Prompt Chaining", same seed
probs = problems[8]
m = model()
SAMPLE = dict(do_sample=True, temperature=0.7, top_p=1.0, top_k=0, repetition_penalty=1.0)
def number(text):
    if "Answer:" not in text:
        return None
    import re
    nums = re.findall(r"-?\d+", text.split("Answer:")[-1].replace(",", ""))
    return int(nums[-1]) if nums else None
torch.manual_seed(0)
answers, tokens = [], []
for start, steps, answer in probs:
    prompt = (f"Start with {start}. " + " ".join(f"Then {s}." for s, _, _ in steps) +
              " Work it out step by step, then give the final result as 'Answer: <number>'.")
    x = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                return_tensors="pt", return_dict=True)["input_ids"].to(DEVICE)
    out = m.generate(x.repeat(16, 1), max_new_tokens=380, **SAMPLE)
    gen = out[:, x.shape[1]:]
    answers.append([number(tok.decode(g, skip_special_tokens=True)) for g in gen])
    tokens.append(int((gen != tok.pad_token_id).sum()) / 16)
    print(f"problem {len(answers)}: {sum(a == answer for a in answers[-1])}/16 samples right", flush=True)
truth = [a for _, _, a in probs]
print(f"\n{len(probs)} problems, eight operations each; {np.mean(tokens):.0f} tokens per sample")
for k in (1, 2, 4, 8, 16):
    maj = [Counter(x for x in a[:k] if x is not None).most_common(1) for a in answers]
    right = np.mean([bool(mc) and mc[0][0] == t for mc, t in zip(maj, truth)])
    some = np.mean([t in a[:k] for a, t in zip(answers, truth)])
    print(f"  majority of {k:2d}: {right:4.0%} right ({k * np.mean(tokens):6.0f} tokens per question); "
          f"some sample right: {some:4.0%}")
per = [sum(x == t for x in a) / 16 for a, t in zip(answers, truth)]
print(f"problems by share of samples right: none {sum(p == 0 for p in per)}, under half {sum(0 < p < 0.5 for p in per)}, "
      f"half or more {sum(0.5 <= p < 1 for p in per)}, all {sum(p == 1 for p in per)}")
