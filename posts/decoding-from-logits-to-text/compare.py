import json
import sys
import time
import torch
from lm import model, tok, encode
from sampling import generate

# Every policy on the same 20 prompts from the blog, 128 new tokens each.
# Measured: how repetitive the text is (the share of 4-token sequences that
# already occurred earlier in the same continuation), and how likely the
# model itself finds it (mean log-probability per token under its plain
# distribution, no temperature or filters). The real continuation of each
# paragraph, written by a person, is scored the same way, over its first 64
# tokens (the shortest is 67), and so are the first 64 of each policy's.
N = 128
STOP = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
prompts = json.load(open("prompts.json"))
POLICIES = {
    "greedy":                 dict(temperature=0),
    "beam search, 4 beams":   "beam",
    "greedy + repetition 1.2": dict(temperature=0, rep=1.2),
    "pure sampling, T=1":     dict(temperature=1.0),
    "T=0.7":                  dict(temperature=0.7),
    "top-k 40":               dict(k=40),
    "top-p 0.9":              dict(p=0.9),
    "min-p 0.1":              dict(mp=0.1),
    "T=1.5 alone":            dict(temperature=1.5),
    "T=1.5 + min-p 0.1":      dict(temperature=1.5, mp=0.1),
}

def repeats(ids, n=4):
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return sum(g in set(grams[:i]) for i, g in enumerate(grams)) / max(1, len(grams))

@torch.no_grad()
def logprob(prompt, ids):
    x = torch.cat([prompt, torch.tensor(ids, device=prompt.device)])
    lp = model(x[None]).logits[0, len(prompt) - 1:-1].float().log_softmax(-1)
    return lp.gather(1, x[len(prompt):, None]).mean().item()

@torch.no_grad()
def beam(prompt):
    out = model.generate(prompt[None], max_new_tokens=N, num_beams=4, do_sample=False,
                         repetition_penalty=1.0,     # the model's generation_config.json sets 1.1
                         eos_token_id=list(STOP), pad_token_id=tok.eos_token_id)
    return out[0, len(prompt):].tolist()

if __name__ == "__main__":
    which = sys.argv[1:] or list(POLICIES)
    results = {}
    for name in which:
        policy = POLICIES[name]
        seeds = (0,) if policy == "beam" or policy.get("temperature", 1) == 0 else (0, 1)
        for seed in seeds:
            t0, texts, reps, reps64, lps = time.perf_counter(), [], [], [], []
            for item in prompts:
                prompt = encode(item["prompt"])
                ids = beam(prompt) if policy == "beam" else generate(model, prompt, N, seed=seed, stop=STOP, **policy)
                ids = [i for i in ids if i not in STOP] or [tok.eos_token_id]
                texts.append(tok.decode(ids)); reps.append(repeats(ids)); reps64.append(repeats(ids[:64]))
                lps.append(logprob(prompt, ids[:64]))
            results[f"{name}|{seed}"] = texts
            print(f"{name:24s} seed {seed}: first 64 tokens: repeated 4-grams {sum(reps64) / 20:5.1%}, "
                  f"log-prob {sum(lps) / 20:5.2f}; all 128: repeated {sum(reps) / 20:5.1%}, "
                  f"{sum(r > 0.3 for r in reps):2d}/20 over 30%  ({time.perf_counter() - t0:.0f} s)", flush=True)
    if not sys.argv[1:]:
        reps, lps = [], []
        for item in prompts:
            prompt, ids = encode(item["prompt"]), tok(item["human"]).input_ids[:64]
            reps.append(repeats(ids)); lps.append(logprob(prompt, ids))
        print(f"{'human continuation':24s}       : first 64 tokens: repeated 4-grams {sum(reps) / 20:5.1%}, "
              f"log-prob {sum(lps) / 20:5.2f}")
        json.dump(results, open("compare_texts.json", "w"), indent=1, ensure_ascii=False)