import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# Real inputs for the experiments: run Qwen2.5-0.5B (base) on the CPU over the
# blog's text and record what goes into the MLP down-projection (4,864 -> 896)
# of layers 12 and 22, with its weight. X_L: 4,096 tokens from the start of
# the text (2,048 to calibrate, 2,048 to measure). H_big_L, mag_big_L: 32,768
# tokens from later in the text, kept only as the sums GPTQ and AWQ need,
# H = 2 X^T X and the mean |x| per input channel.
PATH = snapshot_download("Qwen/Qwen2.5-0.5B", revision="060db6499f32faf8b98477b0a26969ef7d8b9987")
tok = AutoTokenizer.from_pretrained(PATH)
m = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float32).eval()
text = open("blog_train.txt").read()
head = tok(text[:60_000], return_tensors="pt").input_ids[0]
tail = tok(text[60_000:], return_tensors="pt").input_ids[0]
for L in (12, 22):
    layer = m.model.layers[L].mlp.down_proj
    seen, H, absum, n = [], np.zeros((4864, 4864)), np.zeros(4864), 0
    hook = layer.register_forward_hook(lambda mod, inp, out: seen.append(inp[0][0].detach()))
    with torch.no_grad():
        for s in range(0, 4096, 512):
            m(head[s:s + 512][None])
    hook.remove()
    np.save(f"X_{L}.npy", torch.cat(seen).numpy()); np.save(f"W_{L}.npy", layer.weight.detach().numpy())
    def big(mod, inp, out):
        global H, absum, n
        x = inp[0][0].double().numpy()
        H += 2 * x.T @ x; absum += np.abs(x).sum(0); n += len(x)
    hook = layer.register_forward_hook(big)
    with torch.no_grad():
        for s in range(0, 32768, 512):
            m(tail[s:s + 512][None])
    hook.remove()
    np.save(f"H_big_{L}.npy", H); np.save(f"mag_big_{L}.npy", absum / n)
    print(f"layer {L}: inputs {torch.cat(seen).shape[0]} x 4864, calibration tokens {n}")