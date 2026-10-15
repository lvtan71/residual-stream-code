import sys
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download

# Is a full fine-tune's update low-rank? "Fine-Tuning, and What It Destroys"'s full fine-tune saved its
# Transformer-block weights; for every linear layer, take the update
# dW = W_finetuned - W_pretrained and its singular values s, and ask what
# share of its squared size, sum(s_i^2), the best rank-k matrix captures
# (Eckart-Young, as in the widget). For scale, the same for a random
# Gaussian matrix of the same shape, which has no low-rank structure.
PATH = snapshot_download("Qwen/Qwen2.5-0.5B",
                         revision="060db6499f32faf8b98477b0a26969ef7d8b9987")
base = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32).state_dict()
tuned = torch.load(sys.argv[1])
NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
KS = (1, 8, 64)
share = lambda s, k: float((s[:k] ** 2).sum() / (s ** 2).sum())
torch.manual_seed(0)
print(f"update from post 25's full fine-tune ({sys.argv[2]}), averaged over the 24 blocks;")
print("share of the update's squared size captured by the best rank-k matrix, and the rank needed for 90%\n")
print(f"{'matrix':10s} {'shape':>10s}   " + "   ".join(f"rank {k:<3d}" for k in KS) + "   rank for 90%   random matrix, rank 8 / 64")
for n in NAMES:
    keys = [k for k in tuned if k.endswith(f"{n}.weight")]
    rows = []
    for k in keys:
        s = torch.linalg.svdvals(tuned[k].double() - base[k].double())
        need = int((torch.cumsum(s ** 2, 0) / (s ** 2).sum() < 0.9).sum()) + 1
        rows.append([share(s, k_) for k_ in KS] + [need])
    shape = tuple(tuned[keys[0]].shape)
    r = torch.linalg.svdvals(torch.randn(shape, dtype=torch.float64))
    avg = torch.tensor(rows).mean(0)
    print(f"{n:10s} {shape[0]:>4d}x{shape[1]:<5d}   " + "   ".join(f"{x:8.1%}" for x in avg[:3]) +
          f"   {avg[3]:6.0f} of {min(shape)}   {share(r, 8):6.1%} / {share(r, 64):5.1%}")
