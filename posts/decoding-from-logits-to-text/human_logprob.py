import json
import torch
from lm import model, tok, encode

# The real continuations' log-probability, prompt by prompt: how much it
# varies, and whether the paragraphs full of numbers are the hard ones.
prompts = json.load(open("prompts.json"))
rows = []
with torch.no_grad():
    for i, item in enumerate(prompts):
        prompt, ids = encode(item["prompt"]), tok(item["human"]).input_ids[:64]
        x = torch.cat([prompt, torch.tensor(ids, device=prompt.device)])
        lp = model(x[None]).logits[0, len(prompt) - 1:-1].float().log_softmax(-1)
        score = lp.gather(1, x[len(prompt):, None]).mean().item()
        numeric = sum(any(c.isdigit() for c in tok.decode([t])) for t in ids) / len(ids)
        rows.append((score, numeric, i))
rows.sort()
print("per-prompt log-prob of the real continuation: " + ", ".join(f"{s:.2f}" for s, _, _ in rows))
print(f"lowest {rows[0][0]:.2f} (prompt {rows[0][2]}, {rows[0][1]:.0%} of tokens with digits); highest {rows[-1][0]:.2f} "
      f"(prompt {rows[-1][2]}, {rows[-1][1]:.0%} with digits); mean {sum(r[0] for r in rows) / 20:.2f}")