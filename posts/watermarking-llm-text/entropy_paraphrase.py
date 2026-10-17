import numpy as np
import torch
# Two follow-ups to wm.py at delta = 2, regenerating the same texts (same
# prompts and seeds): (1) the green rate per step, grouped by that step's
# own entropy; (2) how much each paraphrase copies the original's token pairs.
src = open("wm.py").read()
exec(src.split("texts, table = {}, {}")[0])                       # model, green(), z_score(), generate(), prompts

@torch.no_grad()
def generate_logged(prompt, delta, temp=0.7, seed=0):
    """generate() from wm.py, also recording each step's entropy (unbiased,
    tempered distribution) and whether the chosen token was green."""
    torch.manual_seed(seed)
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to(DEVICE)
    out = model(ids, use_cache=True)
    past, logits, prev = out.past_key_values, out.logits[0, -1].float().cpu(), ids[0, -1].item()
    toks, steps = [prev], []
    for _ in range(N):
        logp = (logits / temp).log_softmax(-1)
        g = green(prev)
        t = torch.multinomial((logits / temp + delta * g.float()).softmax(-1), 1).item()
        steps.append((-(logp.exp() * logp).sum().item(), bool(g[t])))
        toks.append(t); prev = t
        out = model(torch.tensor([[t]], device=DEVICE), past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1].float().cpu()
    return toks, steps

steps, texts = [], []
for i, p in enumerate(prompts):
    toks, s = generate_logged(p, 2.0, seed=i)
    texts.append(toks); steps += s
ent, grn = np.array([e for e, _ in steps]), np.array([g for _, g in steps])
print(f"delta = 2: {len(steps)} steps, green rate by the step's entropy (nats)")
for lo, hi in ((0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)):
    m = (ent >= lo) & (ent < hi)
    print(f"  entropy {lo:3.1f} to {hi if hi < 99 else 'max':>3}: {m.sum():5d} steps ({m.mean():4.0%}), green {grn[m].mean():.0%}")

print("\nparaphrases: share of their token pairs that also occur in the original")
for i in range(10):
    text = tok.decode(texts[i][1:])
    para, _, _ = generate("Paraphrase the following text, keeping its meaning but changing the wording:\n\n" + text,
                          0.0, seed=100 + i)
    orig = set(zip(texts[i][:-1], texts[i][1:]))
    pp = list(zip(para[:-1], para[1:]))
    print(f"  text {i}: {np.mean([q in orig for q in pp]):.0%} shared, z {z_score(para)[0]:4.1f}, "
          f"starts {tok.decode(para[1:8])!r}", flush=True)