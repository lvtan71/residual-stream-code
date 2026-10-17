import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from common import STUDENT, DEVICE, B, batches, bits_per_byte, evals, gap_to_teacher

# The student, Qwen2.5-0.5B, trained on the same batches three ways:
#   hard: the text's next tokens, ordinary cross-entropy ("Fine-Tuning, and What It Destroys");
#   soft: the teacher's distribution, cross-entropy against its top-64
#         probabilities renormalized (forward KL to that truncated
#         teacher, up to a constant), temperature 1;
#   mix:  the average of the two.
mode, lr, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
trace = len(sys.argv) > 4                                       # also track the distance to the teacher
tv = torch.load("teacher_val.pt") if trace else None
checkpoints = {0, 50, 100, 200, 400}
teacher = torch.load(f"teacher_s{seed}.pt")
torch.manual_seed(seed)
model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float32).to(DEVICE)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()                          # checkpointing needs a grad path past it
model.get_input_embeddings().weight.requires_grad_(False)   # the tied embedding/output table (136M of 494M
                                                              # weights) is frozen, so its gradients and AdamW
                                                              # state (1.6 GB) fit this 16 GB laptop
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
log, t0 = {}, time.perf_counter()
for step, x in enumerate(batches(seed) + [None]):
    if step in checkpoints:
        log[step] = {k: bits_per_byte(model, *v) for k, v in evals.items()}
        if trace:
            log[step]["gap"] = gap_to_teacher(model, tv)
    if x is None:
        break
    x = x.to(DEVICE)
    for group in opt.param_groups:
        group["lr"] = lr * min(1.0, (step + 1) / 20)
    opt.zero_grad(set_to_none=True)
    for rows in (slice(0, B // 2), slice(B // 2, B)):               # two micro-batches: half the memory
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            logits = model(x[rows, :-1]).logits
        lp = logits.float().log_softmax(-1)
        hard = F.nll_loss(lp.reshape(-1, lp.shape[-1]), x[rows, 1:].reshape(-1))
        p = teacher["lp"][step][rows].float().to(DEVICE).softmax(-1)  # top-64, renormalized
        soft = -(p * lp.gather(-1, teacher["id"][step][rows].long().to(DEVICE))).sum(-1).mean()
        loss = {"hard": hard, "soft": soft, "mix": 0.5 * (hard + soft)}[mode] / 2
        loss.backward()
        del logits, lp, loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
print(f"{mode:4s} lr {lr:g} seed {seed} ({time.perf_counter() - t0:.0f} s): " +
      " | ".join(f"step {s}: blog {v['blog held-out']:.3f}, wiki {v['WikiText-2']:.3f}" + (f", KL to teacher {v['gap']:.3f}" if trace else "") for s, v in log.items()), flush=True)