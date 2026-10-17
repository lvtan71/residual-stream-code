# The first step's logits for four prompts, alone and in a padded batch,
# in 16-bit and in 32-bit floats (run once with each: python det_check.py 16 / 32).
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm import PATH
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = {"16": torch.float16, "32": torch.float32}[sys.argv[1]]
tok = AutoTokenizer.from_pretrained(PATH); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=dtype).to(DEVICE).eval()
prompts = [p["prompt"] for p in json.load(open("prompts.json"))][:4]
with torch.no_grad():
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    b = model(**enc).logits[:, -1].float()
    b2 = model(**enc, position_ids=pos).logits[:, -1].float()
    for j, p in enumerate(prompts):
        a = model(**tok([p], return_tensors="pt").to(DEVICE)).logits[0, -1].float()
        print(sys.argv[1], j, "pad tokens", int((enc.attention_mask[j] == 0).sum()), "diff", round((a - b[j]).abs().max().item(), 4), "with position_ids", round((a - b2[j]).abs().max().item(), 4))