import bz2
import math
import sys
import torch
import torch.nn.functional as F
from gpt import GPT, Config
from repeats import bits_each
from train import final_split, tokenizer_for
from kn import build, prob                        # Kneser-Ney from "Sequences Before Attention", unchanged
from lm import LSTMLM                              # LSTM from "What 'Generative' Means"

# The two held-out posts scored separately, by every model on the ladder in
# "What 'Generative' Means", plus the GPT's score on training text it has seen, at each saved
# checkpoint (seed 0). Each model scores the whole held-out text as one
# piece, as before; the per-character costs are then summed per post.
torch.set_num_threads(4)
train, val = final_split["fit"], final_split["score"]
cut = val.find("When an LLM generates text")       # the KV-cache post starts here
posts = {"BPE post": (1, cut), "KV-cache post": (cut, len(val))}
nbytes = lambda s: len(s.encode("utf-8"))
_, encode, vocab = tokenizer_for("char", train)

def per_post(bits):                                 # bits[i] is the cost of character i
    return {name: bits[a:b].sum().item() / nbytes(val[a:b]) for name, (a, b) in posts.items()}

unseen = torch.tensor([c not in set(train) for c in val])
print(f"{unseen.sum()} held-out characters never occur in training: "
      + ", ".join(f"{name} {unseen[a:b].sum()}" for name, (a, b) in posts.items()))

rows = {}
for step in (250, 1000, 2000, 3000, 4000):
    ckpt = torch.load(f"gpt_{step}.pt", weights_only=False)
    model = GPT(Config(**ckpt["config"])); model.load_state_dict(ckpt["model"]); model.eval()
    bits = torch.cat([torch.zeros(1), bits_each(model, encode(val), model.c.T)])
    seen_text = train[:60_000]
    train_bpb = bits_each(model, encode(seen_text), model.c.T).sum().item() / nbytes(seen_text[1:])
    rows[f"GPT, {step:,} steps"] = (per_post(bits), train_bpb)
    if step == 4000:
        cost = bits[unseen].sum().item()
        print(f"the GPT at 4,000 steps pays {cost / unseen.sum():.1f} bits per unseen character, "
              f"{cost / nbytes(val):.3f} bits per byte of the held-out text in all")

lstm = torch.load("char_lstm.pt", weights_only=False)
index = {u: i + 1 for i, u in enumerate(lstm["vocab"])}
m = LSTMLM(len(lstm["vocab"]) + 1); m.load_state_dict(lstm["model"]); m.eval()
data, bits, state = torch.tensor([index.get(c, 0) for c in val]), torch.zeros(len(val)), None
with torch.no_grad():
    for s in range(0, len(data) - 1, 128):
        x, y = data[s:s + 128], data[s + 1:s + 129]
        logits, state = m(x[None][:, :len(y)], state)
        bits[s + 1:s + 1 + len(y)] = F.cross_entropy(logits[0], y, reduction="none") / math.log(2)
rows["LSTM (post 9, seed 0)"] = (per_post(bits), None)

tables = build(6)
bits = torch.tensor([0.0] + [-math.log2(prob(tables, val[max(0, i - 6):i], val[i])) for i in range(1, len(val))])
rows["Kneser-Ney, 6 characters"] = (per_post(bits), None)

tb = train.encode()
rows["bz2 given the training posts"] = ({name: 8 * (len(bz2.compress(tb + val[a:b].encode(), 9)) - len(bz2.compress(tb, 9)))
                                         / nbytes(val[a:b]) for name, (a, b) in posts.items()}, None)

print(f"\n{'bits per byte':30s} {'BPE post':>9s} {'KV-cache post':>14s} {'training text (seen)':>21s}")
for name, (pp, tr) in rows.items():
    print(f"{name:30s} {pp['BPE post']:9.3f} {pp['KV-cache post']:14.3f} "
          + (f"{tr:21.3f}" if tr is not None else ""))