import json

# One prompt's continuations from compare.py, under each policy (seed 0),
# first 260 characters, with the paragraph's real continuation last.
texts, prompts = json.load(open("compare_texts.json")), json.load(open("prompts.json"))
i = 7
print(f"PROMPT: {prompts[i]['prompt']}\n")
for name in ("greedy", "beam search, 4 beams", "greedy + repetition 1.2", "pure sampling, T=1", "top-p 0.9",
             "min-p 0.1", "T=1.5 alone", "T=1.5 + min-p 0.1"):
    print(f"[{name}]{texts[name + '|0'][i][:260]!r}\n")
print(f"[what the post says]{prompts[i]['human'][:260]!r}")