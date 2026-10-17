import math
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# Which part of the full fine-tune's update does the work? Rebuild the
# model with each matrix's update cut to its best rank-k approximation,
# W_pretrained + (dW)_k, everything else (biases, norm gains) as fine-tuned,
# and measure the blog and WikiText as before. k = 0 keeps only the
# biases and norm gains; "all" is the fine-tuned model itself, except the
# final norm (not saved by ft.py), which stays pretrained.
PATH = snapshot_download("Qwen/Qwen2.5-0.5B",
                         revision="060db6499f32faf8b98477b0a26969ef7d8b9987")
WIKI = snapshot_download("Salesforce/wikitext", repo_type="dataset",
                         revision="b08601e04326c79dfdd32d625aee71d232d685c3") + "/wikitext-2-raw-v1/"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
T = 512
NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
tok = AutoTokenizer.from_pretrained(PATH)
wiki_test = "".join(pd.read_parquet(WIKI + "test-00000-of-00001.parquet")["text"])[:120_000]
blog_val = open("blog_val.txt").read()
evals = {"blog": (torch.tensor(tok(blog_val).input_ids), blog_val),
         "wiki": (torch.tensor(tok(wiki_test).input_ids), wiki_test)}
model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32).to(DEVICE).eval()
base = {k: v.clone().cpu() for k, v in model.state_dict().items()}
tuned = torch.load("ft_weights.pt")
svd = {k: torch.linalg.svd(tuned[k].double() - base[k].double(), full_matrices=False)
       for k in tuned if any(k.endswith(f"{n}.weight") for n in NAMES)}

@torch.no_grad()
def bits_per_byte(data, text):
    total = 0.0
    for s in range(0, len(data) - 1, T):
        x = data[s:s + T + 1].to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            logits = model(x[None, :-1]).logits[0].float()
        total += F.cross_entropy(logits, x[1:], reduction="sum").item()
    return total / math.log(2) / len(text.encode())

print("full fine-tune's update cut to rank k in every block matrix (bits per byte)")
for k in ("pretrained", 0, 1, 8, 64, "all"):
    sd = dict(base)
    if k != "pretrained":
        sd.update(tuned)                                            # fine-tuned biases, norms, matrices
        if k != "all":
            for name, (U, S, Vh) in svd.items():
                sd[name] = (base[name].double() + (U[:, :k] * S[:k]) @ Vh[:k]).float()
    model.load_state_dict({n: v.to(DEVICE) for n, v in sd.items()})
    r = {n: bits_per_byte(*v) for n, v in evals.items()}
    print(f"  {str(k):>10s}: blog {r['blog']:.3f}, wiki {r['wiki']:.3f}", flush=True)