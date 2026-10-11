import json
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# Causal attention in a real decoder, Qwen2.5-1.5B-Instruct (28 layers, 12
# query heads each). First, the mask: change the last token of a sentence and
# check that no earlier position's output moves. Then what the heads learned:
# for every head, the average weight a token puts on the token just before it,
# and on the sentence's first token; the strongest of each kind is saved to
# heads.json for the post's figures. Attention weights need the plain
# ("eager") attention code, so the model is loaded with it here.
PATH = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32, attn_implementation="eager").to(DEVICE).eval()
text = "The cat sat on the mat because the mat was warm and the cat was cold."
ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
other = ids.clone()
other[0, -1] = tok(" hot").input_ids[0]                      # a different last token

with torch.no_grad():
    a = model(ids, output_hidden_states=True, output_attentions=True)
    b = model(other, output_hidden_states=True)
n = ids.shape[1]
before = max((x[0, :-1] - y[0, :-1]).abs().max().item() for x, y in zip(a.hidden_states, b.hidden_states))
last = (a.hidden_states[-1][0, -1] - b.hidden_states[-1][0, -1]).abs().max().item()
print(f"{n} tokens; changing the last one moves earlier positions' hidden states by at most {before:.1e} "
      f"(in every layer), and the last position's by {last:.1f}")

att = torch.stack(a.attentions)[:, 0]                       # [layers, heads, n, n]
above = att.triu(1).abs().max().item()
print(f"largest attention weight above the diagonal, any layer or head: {above:.1e}")
prev = att.diagonal(-1, -2, -1).mean(-1)                     # weight on the token just before, averaged
first = att[..., 1:, 0].mean(-1)                             # weight on the first token, rows 1..n-1
print(f"average over all heads: weight on the previous token {prev.mean():.2f}, on the first token {first.mean():.2f}")
picks = {}
for name, score in (("previous token", prev), ("first token", first)):
    layer, head = divmod(score.flatten().argmax().item(), score.shape[1])
    picks[name] = {"layer": layer, "head": head, "weights": att[layer, head].tolist()}
    print(f"strongest {name} head: layer {layer}, head {head}, average weight {score[layer, head]:.2f}")
picks["tokens"] = [tok.decode(t) for t in ids[0]]
json.dump(picks, open("heads.json", "w"))
