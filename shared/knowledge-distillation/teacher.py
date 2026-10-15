import sys
import torch
from transformers import AutoModelForCausalLM
from common import TEACHER, DEVICE, TOPK, batches, bits_per_byte, evals

# The teacher, Qwen2.5-1.5B-Instruct, reads every training batch once; its
# top-64 next-token log-probabilities at every position are saved, which is
# what the student is distilled from. Also: the teacher's own bits per byte.
seed = int(sys.argv[1])
model = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.float16).to(DEVICE).eval()
top_lp, top_id, kept = [], [], []
with torch.no_grad():
    for x in batches(seed):
        lp = model(x[:, :-1].to(DEVICE)).logits.float().log_softmax(-1)
        v, i = lp.topk(TOPK, -1)
        top_lp.append(v.half().cpu()); top_id.append(i.int().cpu())
        kept.append(v.exp().sum(-1).mean().item())
torch.save({"lp": torch.stack(top_lp), "id": torch.stack(top_id)}, f"teacher_s{seed}.pt")
if seed == 0:
    scores = {k: bits_per_byte(model, *v) for k, v in evals.items()}
    print(f"teacher: blog {scores['blog held-out']:.3f}, wiki {scores['WikiText-2']:.3f} bits per byte; "
          f"its top {TOPK} tokens hold {sum(kept) / len(kept):.1%} of its probability on average", flush=True)
