import copy
import time
import numpy as np
import torch
from lm import model, tok, DEVICE
sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(DEVICE, lambda: None)   # the post ran on "mps"

# What a cached prefix saves, measured with Qwen2.5-1.5B-Instruct on this
# laptop. A long shared prefix (instructions plus reference text, here text
# from this blog's posts) is followed by a short question.
#   cold:   the model processes prefix + question from scratch (prefill);
#   cached: the prefix's KV cache was computed once and kept; only the
#           question's tokens are processed, attending to it.
# Time to the first output token, median of 5, for prefixes of 1k to 8k.
m = model()
text = open("blog_train.txt").read()
question = tok("\n\nQuestion: what does this text say about the KV cache? Answer:", return_tensors="pt").input_ids.to(DEVICE)
cfg = m.config
kv_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2
print(f"KV cache per token: {kv_per_token / 1024:.0f} KiB (2 x {cfg.num_hidden_layers} layers x "
      f"{cfg.num_key_value_heads} KV heads x {cfg.hidden_size // cfg.num_attention_heads} x 2 bytes)")

@torch.no_grad()
def first_token(ids, past=None):
    sync(); t0 = time.perf_counter()
    out = m(ids, past_key_values=past, use_cache=True, logits_to_keep=1)   # logits for the last position only, as a server does
    out.logits[0, -1].argmax().item()                                 # forces the result
    sync()
    return time.perf_counter() - t0, out

print(f"{'prefix':>7s} {'cold':>9s} {'cached':>9s} {'saved':>6s} {'prefix KV held':>15s}")
for n in (1024, 2048, 4096, 8192):
    prefix = tok(text[:n * 6], return_tensors="pt").input_ids[:, :n].to(DEVICE)
    full = torch.cat([prefix, question], 1)
    first_token(full)                                                 # warm-up
    cold = np.median([first_token(full)[0] for _ in range(5)])
    _, pre = first_token(prefix)                                      # compute the prefix's cache once
    cached = np.median([first_token(question, copy.deepcopy(pre.past_key_values))[0] for _ in range(5)])
    print(f"{n:7d} {cold * 1e3:7.0f} ms {cached * 1e3:7.0f} ms {1 - cached / cold:6.0%} {n * kv_per_token / 2**20:11.0f} MiB")
