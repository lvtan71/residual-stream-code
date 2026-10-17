import json
import torch
from lm import model, tok, encode, DEVICE
from sampling import generate

# Is decoding reproducible? (1) The same seed twice. (2) The same prompt
# decoded greedily on its own, and inside a batch of four prompts padded to
# the same length: mathematically the same computation for this prompt.
prompts = [p["prompt"] for p in json.load(open("prompts.json"))]

same = sum(generate(model, encode(p), 64, seed=7, p=0.9) == generate(model, encode(p), 64, seed=7, p=0.9)
           for p in prompts[:5])
print(f"top-p 0.9, the same seed twice: identical text for {same} of 5 prompts")

@torch.no_grad()
def greedy(batch):
    tok.padding_side = "left"
    enc = tok(batch, return_tensors="pt", padding=True).to(DEVICE)
    out = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id,
                         repetition_penalty=1.0,     # the model's generation_config.json sets 1.1
                         output_scores=True, return_dict_in_generate=True)
    return out.sequences[:, enc.input_ids.shape[1]:].tolist(), out.scores[0].float()

diverged, first, gaps = 0, [], []
for i in range(0, 20, 4):
    batch = prompts[i:i + 4]
    together, scores_b = greedy(batch)
    for j, p in enumerate(batch):
        alone, scores_a = greedy([p])
        gaps.append((scores_a[0] - scores_b[j]).abs().max().item())
        a, b = alone[0], together[j]
        if a != b:
            diverged += 1
            first.append(next(k for k in range(min(len(a), len(b))) if a[k] != b[k]))
print(f"greedy, alone vs in a padded batch of 4: largest first-step logit difference "
      f"{max(gaps):.3f} (median {sorted(gaps)[10]:.3f}); the 64-token outputs differ for {diverged} of 20 "
      f"prompts, first at token {sorted(first) if first else '-'}")