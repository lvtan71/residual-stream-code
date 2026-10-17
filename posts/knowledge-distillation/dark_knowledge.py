import torch
from transformers import AutoModelForCausalLM
from common import TEACHER, DEVICE, tok

# "Dark knowledge": what a teacher's distribution says beyond its top
# answer. The teacher's next-token probabilities after one sentence of this
# blog, at temperature 1 and softened at temperature 2. A hard label keeps
# only the token the text happened to use.
model = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.float32).to(DEVICE).eval()
text = "A language model generates its answer one"
with torch.no_grad():
    logits = model(tok(text, return_tensors="pt").input_ids.to(DEVICE)).logits[0, -1].float()
print(repr(text))
for tau in (1, 2):
    p = (logits / tau).softmax(-1)
    v, i = p.topk(8)
    print(f"temperature {tau}: " + ", ".join(f"{tok.decode(int(t))!r} {float(x):.3f}" for x, t in zip(v, i)) +
          f"; top 8 hold {float(v.sum()):.0%}")