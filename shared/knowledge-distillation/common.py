import math
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # notebook: the post ran on "mps"

# Shared by the teacher and student scripts: post 25's data, batches and
# bits-per-byte measurement.
STUDENT = snapshot_download("Qwen/Qwen2.5-0.5B",
                            revision="060db6499f32faf8b98477b0a26969ef7d8b9987")
TEACHER = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                            revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
WIKI = snapshot_download("Salesforce/wikitext", repo_type="dataset",
                         revision="b08601e04326c79dfdd32d625aee71d232d685c3") + "/wikitext-2-raw-v1/"
DEVICE, T, B, STEPS, TOPK = GPU, 512, 4, 400, 64

tok = AutoTokenizer.from_pretrained(STUDENT)                       # the teacher's vocabulary is identical
wiki_test = "".join(pq.read_table(WIKI + "test-00000-of-00001.parquet")["text"].to_pylist())[:120_000]
blog_train, blog_val = open("blog_train.txt").read(), open("blog_val.txt").read()
ids = lambda s: torch.tensor(tok(s).input_ids)
blog_ids = ids(blog_train)
evals = {"blog held-out": (ids(blog_val), blog_val), "WikiText-2": (ids(wiki_test), wiki_test)}

def batches(seed):
    """The STEPS training batches for a seed, fixed, so teacher and student see the same ones."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(STEPS):
        starts = [torch.randint(0, len(blog_ids) - T - 1, (1,), generator=g).item() for _ in range(B)]
        out.append(torch.stack([blog_ids[i:i + T + 1] for i in starts]))
    return out

@torch.no_grad()
def bits_per_byte(model, data, text):
    model.eval()
    total = 0.0
    for s in range(0, len(data) - 1, T):
        x = data[s:s + T + 1].to(DEVICE)
        with torch.autocast(GPU, dtype=torch.bfloat16):
            logits = model(x[None, :-1]).logits[0].float()
        total += F.cross_entropy(logits, x[1:], reduction="sum").item()
    model.train()
    return total / math.log(2) / len(text.encode())

@torch.no_grad()
def gap_to_teacher(model, tv):
    """On the held-out blog windows: KL from the teacher to the student over the
    teacher's top 64 tokens, both renormalized to them, averaged over positions (nats)."""
    model.eval()
    data, total, n = evals["blog held-out"][0], 0.0, 0
    for k, s in enumerate(range(0, len(data) - 1, T)):
        x = data[s:s + T + 1].to(DEVICE)
        with torch.autocast(GPU, dtype=torch.bfloat16):
            lq = model(x[None, :-1]).logits[0].float().log_softmax(-1)
        lp = tv["lp"][k].float().to(DEVICE).log_softmax(-1)          # top 64, renormalized
        lq = lq.gather(-1, tv["id"][k].long().to(DEVICE)).log_softmax(-1)        # the student, on the same 64
        total += (lp.exp() * (lp - lq)).sum().item(); n += lp.shape[0]
    model.train()
    return total / n
