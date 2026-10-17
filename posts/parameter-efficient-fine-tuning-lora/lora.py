import math
import sys
import time
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# LoRA on "Fine-Tuning, and What It Destroys"'s fine-tune: the same model (Qwen2.5-0.5B base), data,
# schedule and measurements, but every pretrained weight frozen. Each linear
# layer W (d_out x d_in) in the Transformer blocks gets a trainable low-rank
# update, W x + (alpha / r) B A x, with A (r x d_in) small and random and
# B (d_out x r) zero, so training starts from exactly the pretrained model.
PATH = snapshot_download("Qwen/Qwen2.5-0.5B",
                         revision="060db6499f32faf8b98477b0a26969ef7d8b9987")
WIKI = snapshot_download("Salesforce/wikitext", repo_type="dataset",
                         revision="b08601e04326c79dfdd32d625aee71d232d685c3") + "/wikitext-2-raw-v1/"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
T, B, STEPS = 512, 4, 400
rank, lr, seed = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else STEPS          # a short smoke run
alpha = 2 * rank                                   # alpha / r = 2 at every rank
checkpoints = {0, 25, 50, 100, 200, 400, STEPS}

class LoRALinear(torch.nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T
    def merged(self):
        """W + (alpha / r) B A: one ordinary matrix again, no extra cost at inference."""
        return self.base.weight + self.scale * self.B @ self.A

tok = AutoTokenizer.from_pretrained(PATH)
wiki_test = "".join(pd.read_parquet(WIKI + "test-00000-of-00001.parquet")["text"])[:120_000]
blog_train, blog_val = open("blog_train.txt").read(), open("blog_val.txt").read()
ids = lambda s: torch.tensor(tok(s).input_ids)
blog_ids = ids(blog_train)
evals = {"blog held-out": (ids(blog_val), blog_val), "WikiText-2": (ids(wiki_test), wiki_test)}

torch.manual_seed(seed)
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32).to(DEVICE)
for p in model.parameters():
    p.requires_grad_(False)
names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
for layer in model.model.layers:
    for parent in (layer.self_attn, layer.mlp):
        for n in names:
            if hasattr(parent, n):
                setattr(parent, n, LoRALinear(getattr(parent, n), rank, alpha).to(DEVICE))
model.gradient_checkpointing_enable()
model.enable_input_require_grads()                # checkpointing needs a grad path into the frozen model
trainable = [p for p in model.parameters() if p.requires_grad]
n_train, n_all = sum(p.numel() for p in trainable), sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
g = torch.Generator().manual_seed(seed)

@torch.no_grad()
def bits_per_byte(data, text):
    model.eval()
    total = 0.0
    for s in range(0, len(data) - 1, T):
        x = data[s:s + T + 1].to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            logits = model(x[None, :-1]).logits[0].float()
        total += F.cross_entropy(logits, x[1:], reduction="sum").item()
    model.train()
    return total / math.log(2) / len(text.encode())

log = {}
t0 = time.perf_counter()
for step in range(STEPS + 1):
    if step in checkpoints:
        log[step] = {k: bits_per_byte(*v) for k, v in evals.items()}
    if step == STEPS:
        break
    rows = []
    for _ in range(B):
        i = torch.randint(0, len(blog_ids) - T - 1, (1,), generator=g).item()
        rows.append(blog_ids[i:i + T + 1])
    x = torch.stack(rows).to(DEVICE)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        logits = model(x[:, :-1]).logits
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
    for group in opt.param_groups:
        group["lr"] = lr * min(1.0, (step + 1) / 20)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
seconds = time.perf_counter() - t0

# Merging: fold B A into W and check the merged model computes the same thing.
x = blog_ids[:T][None].to(DEVICE)
with torch.no_grad():
    before = model(x).logits.float()
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for n in names:
                m = getattr(parent, n, None)
                if isinstance(m, LoRALinear):
                    m.base.weight.copy_(m.merged()); setattr(parent, n, m.base)
    after = model(x).logits.float()
print(f"rank {rank:2d}, lr {lr:g}, seed {seed}: {n_train / 1e6:.2f}M trainable of {n_all / 1e6:.0f}M "
      f"({n_train / n_all:.2%}), {seconds:.0f} s | " +
      " | ".join(f"step {s}: blog {v['blog held-out']:.3f}, wiki {v['WikiText-2']:.3f}" for s, v in log.items()) +
      f" | merged vs unmerged logits, max difference {(before - after).abs().max().item():.1e}", flush=True)