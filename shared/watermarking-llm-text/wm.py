import math
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# The Kirchenbauer et al. (2023) watermark on a real model. At each step,
# the previous token and a secret key seed a random split of the vocabulary:
# a "green" fraction gamma and the rest red. delta is added to the green
# tokens' logits before sampling. Detection needs only the key and the
# tokenizer: count green tokens and compute a z-score.
PATH = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                         revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
KEY, GAMMA, N = 15485863, 0.25, 200
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16).to(DEVICE).eval()
V = model.config.vocab_size
_green = {}
def green(prev):
    """The green list for a given previous token: a keyed permutation of the
    vocabulary, of which the first gamma V tokens are green."""
    if prev not in _green:
        g = torch.Generator().manual_seed(KEY * (prev + 1) % (2 ** 63 - 1))
        mask = torch.zeros(V, dtype=torch.bool)
        mask[torch.randperm(V, generator=g)[: int(GAMMA * V)]] = True
        _green[prev] = mask
    return _green[prev]

def z_score(ids):
    """ids: the scored tokens, preceded by the one token before them."""
    hits = sum(bool(green(p)[t]) for p, t in zip(ids[:-1], ids[1:]))
    n = len(ids) - 1
    return (hits - GAMMA * n) / math.sqrt(n * GAMMA * (1 - GAMMA)), hits / n

@torch.no_grad()
def generate(prompt, delta, temp=0.7, seed=0):
    """Sample N tokens with the watermark; also return each token's
    log-probability and the entropy under the unwatermarked distribution."""
    torch.manual_seed(seed)
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to(DEVICE)
    out = model(ids, use_cache=True)
    past, logits, prev = out.past_key_values, out.logits[0, -1].float().cpu(), ids[0, -1].item()
    toks, nll, ent = [prev], [], []
    for _ in range(N):
        logp = (logits / temp).log_softmax(-1)
        biased = logits / temp + delta * green(prev).float()
        t = torch.multinomial(biased.softmax(-1), 1).item()
        nll.append(-logp[t].item()); ent.append(-(logp.exp() * logp).sum().item())
        toks.append(t); prev = t
        out = model(torch.tensor([[t]], device=DEVICE), past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1].float().cpu()
    return toks, float(np.mean(nll)), float(np.mean(ent))

prompts = ["Write a short essay about why cities have parks.", "Describe a morning at a fishing village.",
           "Explain how a bicycle stays upright.", "Write a story about a lost umbrella.",
           "What makes a good teacher?", "Describe the history of bread in a few paragraphs.",
           "Write a letter to a friend about learning to cook.", "Explain why the sky changes color at sunset.",
           "Write a product description for a wooden desk lamp.", "Describe a city at night from a rooftop."]
texts, table = {}, {}
print(f"gamma = {GAMMA}, {N} tokens per text, temperature 0.7, 10 prompts")
print(f"{'delta':>5s} {'green share':>12s} {'z (mean, min)':>15s} {'detected z>4':>13s} {'NLL (nats/token)':>17s} {'entropy':>8s}")
for delta in (0.0, 1.0, 2.0, 4.0):
    rows = []
    for i, p in enumerate(prompts):
        toks, nll, ent = generate(p, delta, seed=i)
        z, g = z_score(toks)
        rows.append((z, g, nll, ent)); texts[(delta, i)] = toks
    table[delta] = rows
    z, g, nll, ent = map(np.array, zip(*rows))
    print(f"{delta:5.0f} {g.mean():12.3f} {z.mean():8.1f} {z.min():5.1f} {(z > 4).sum():9d}/10 {nll.mean():17.3f} {ent.mean():8.2f}", flush=True)

# The watermark needs uncertainty to work with: at delta = 2, each text's
# z against the model's average entropy while writing it.
z2, ent2 = np.array([r[0] for r in table[2.0]]), np.array([r[3] for r in table[2.0]])
order = np.argsort(ent2)
print("\ndelta = 2, texts from lowest to highest entropy:")
print("  entropy " + " ".join(f"{v:5.2f}" for v in ent2[order]))
print("  z       " + " ".join(f"{v:5.1f}" for v in z2[order]))
print(f"  correlation {np.corrcoef(ent2, z2)[0, 1]:.2f}")

# Attacks on the delta = 2 texts.
rng = np.random.default_rng(0)
print(f"\nattacks on the delta = 2 texts: mean z (texts still detected at z > 4)")
for frac in (0.1, 0.2, 0.3, 0.5):
    zz = []
    for i in range(10):
        t = list(texts[(2.0, i)])
        for j in rng.choice(np.arange(1, len(t)), int(frac * N), replace=False):
            t[j] = int(rng.integers(0, len(tok)))          # a random token in place of the original
        zz.append(z_score(t)[0])
    print(f"  replace {frac:.0%} of tokens at random: z {np.mean(zz):5.1f} ({sum(v > 4 for v in zz)}/10)", flush=True)
zz = []
for i in range(10):
    text = tok.decode(texts[(2.0, i)][1:])
    para, _, _ = generate("Paraphrase the following text, keeping its meaning but changing the wording:\n\n" + text,
                          0.0, seed=100 + i)
    zz.append(z_score(para)[0])
print(f"  paraphrase by the same model, unwatermarked: z {np.mean(zz):5.1f} ({sum(v > 4 for v in zz)}/10)")
