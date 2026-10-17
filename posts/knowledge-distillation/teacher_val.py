import torch
from transformers import AutoModelForCausalLM
from common import TEACHER, DEVICE, T, TOPK, evals

# The teacher's top-64 log-probabilities on the held-out blog windows, so a
# student's distance from the teacher can be measured on text neither trained on.
model = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.float16).to(DEVICE).eval()
data, lp, ids = evals["blog held-out"][0], [], []
with torch.no_grad():
    for s in range(0, len(data) - 1, T):
        x = data[s:s + T + 1].to(DEVICE)
        v, i = model(x[None, :-1]).logits[0].float().log_softmax(-1).topk(TOPK, -1)
        lp.append(v.half().cpu()); ids.append(i.int().cpu())
torch.save({"lp": lp, "id": ids}, "teacher_val.pt")
