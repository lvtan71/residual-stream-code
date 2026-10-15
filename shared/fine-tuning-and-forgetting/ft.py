import json
import math
import sys
import time
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # notebook: the post ran on "mps"

# Fine-tuning a pretrained model on new text, and what it forgets. Qwen2.5-
# 0.5B (base) is trained further on this blog's training posts ("What Generative Means"'s
# frozen snapshot), every weight updated except the embedding table (full
# fine-tuning, near enough; see below). Measured along the way, in bits per
# byte: the two held-out blog posts (what it's meant to learn), WikiText-2's
# test set (general English it knew before) and Vietnamese Wikipedia
# (another thing it knew, which no variant trains on). Variants: the learning rate, and "replay", mixing
# general English text (raw Wikipedia, from wiki_data.py) back into each
# batch.
PATH = snapshot_download("Qwen/Qwen2.5-0.5B",
                         revision="060db6499f32faf8b98477b0a26969ef7d8b9987")
WIKI = snapshot_download("Salesforce/wikitext", repo_type="dataset",
                         revision="b08601e04326c79dfdd32d625aee71d232d685c3") + "/wikitext-2-raw-v1/"
DEVICE, T, B = GPU, 512, 4
lr, replay, steps, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
checkpoints = {0, 25, 50, 100, 200, 400} | {steps}

tok = AutoTokenizer.from_pretrained(PATH)
wiki_test = "".join(pd.read_parquet(WIKI + "test-00000-of-00001.parquet")["text"])[:120_000]
blog_train, blog_val = open("blog_train.txt").read(), open("blog_val.txt").read()
replay_text, vi_test = open("wiki_replay.txt").read(), open("vi_test.txt").read()
ids = lambda s: torch.tensor(tok(s).input_ids)
blog_ids, wiki_ids = ids(blog_train), ids(replay_text)
evals = {"blog held-out": (ids(blog_val), blog_val), "WikiText-2": (ids(wiki_test), wiki_test),
         "Vietnamese": (ids(vi_test), vi_test)}

torch.manual_seed(seed)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32).to(DEVICE)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()                          # checkpointing needs a grad path past it
model.get_input_embeddings().weight.requires_grad_(False)   # the tied embedding/output table (136M of 494M
                                                              # weights) is frozen, so its gradients and AdamW
                                                              # state (1.6 GB) fit this 16 GB laptop
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
g = torch.Generator().manual_seed(seed)

@torch.no_grad()
def bits_per_byte(data, text):
    """Non-overlapping windows of T tokens (the same windows for every model)."""
    model.eval()
    total = 0.0
    for s in range(0, len(data) - 1, T):
        x = data[s:s + T + 1].to(DEVICE)
        with torch.autocast(GPU, dtype=torch.bfloat16):
            logits = model(x[None, :-1]).logits[0].float()
        total += F.cross_entropy(logits, x[1:], reduction="sum").item()
    model.train()
    return total / math.log(2) / len(text.encode())

def batch():
    n_wiki = round(B * replay)
    rows = []
    for k in range(B):
        src = wiki_ids if k < n_wiki else blog_ids
        i = torch.randint(0, len(src) - T - 1, (1,), generator=g).item()
        rows.append(src[i:i + T + 1])
    return torch.stack(rows).to(DEVICE)

log = {}
t0 = time.perf_counter()
for step in range(steps + 1):
    if step in checkpoints:
        log[step] = {k: round(bits_per_byte(*v), 4) for k, v in evals.items()}
    if step == steps:
        break
    x = batch()
    for group in opt.param_groups:                                  # 20-step warmup, then constant
        group["lr"] = lr * min(1.0, (step + 1) / 20)
    opt.zero_grad(set_to_none=True)
    for half in (x[:B // 2], x[B // 2:]):                            # two micro-batches of 2: the same batch of 4,
        with torch.autocast(GPU, dtype=torch.bfloat16):           # in half the memory for activations and logits
            logits = model(half[:, :-1]).logits
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), half[:, 1:].reshape(-1)) / 2
        loss.backward()
        del logits, loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
print(f"lr {lr:g}, replay {replay:.0%}, seed {seed} ({time.perf_counter() - t0:.0f} s): " +
      " | ".join(f"step {s}: blog {v['blog held-out']:.3f}, wiki {v['WikiText-2']:.3f}, vi {v['Vietnamese']:.3f}" for s, v in log.items()), flush=True)
if len(sys.argv) > 5:                                                # "Parameter-Efficient Fine-Tuning" measures the update's rank
    torch.save({k: v.cpu() for k, v in model.state_dict().items() if ".layers." in k}, sys.argv[5])
