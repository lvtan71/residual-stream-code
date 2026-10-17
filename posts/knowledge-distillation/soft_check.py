import sys
import torch
from transformers import AutoModelForCausalLM
from common import STUDENT, DEVICE, B, batches, gap_to_teacher

# Does training on soft labels lower its own loss? 100 steps of distill.py's
# soft run, logging the training loss split in two: the KL to the teacher
# over its top 64 tokens (renormalized), and -log of the student's
# probability mass on those 64 (the loss also pulls mass into them).
lr = float(sys.argv[1])
teacher, tv = torch.load("teacher_s0.pt"), torch.load("teacher_val.pt")
torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float32).to(DEVICE)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.get_input_embeddings().weight.requires_grad_(False)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
print(f"lr {lr:g}: held-out KL to teacher {gap_to_teacher(model, tv):.3f}", flush=True)
acc = []
for step, x in enumerate(batches(0)[:100]):
    x = x.to(DEVICE)
    for group in opt.param_groups:
        group["lr"] = lr * min(1.0, (step + 1) / 20)
    opt.zero_grad(set_to_none=True)
    for rows in (slice(0, B // 2), slice(B // 2, B)):
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            logits = model(x[rows, :-1]).logits
        lq = logits.float().log_softmax(-1).gather(-1, teacher["id"][step][rows].long().to(DEVICE))
        lp = teacher["lp"][step][rows].float().to(DEVICE).log_softmax(-1)
        mass = lq.logsumexp(-1)                                         # log of the student's mass on the 64
        kl = (lp.exp() * (lp - (lq - mass[..., None]))).sum(-1).mean()
        soft = -(lp.exp() * lq).sum(-1).mean()
        (soft / 2).backward()
        acc.append((kl.item(), -mass.mean().item()))
        del logits, lq
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (step + 1) % 10 == 0:
        k, m = torch.tensor(acc).mean(0).tolist(); acc = []
        print(f"  steps {step - 8:3d}-{step + 1:3d}: training KL {k:.3f}, -log mass on top 64 {m:.3f}", flush=True)
print(f"  after 100 steps: held-out KL to teacher {gap_to_teacher(model, tv):.3f}", flush=True)