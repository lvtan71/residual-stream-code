import copy
import time
import torch
from transformers import DynamicCache
from lm import model, tok, DEVICE

# What a cached prefix saves, measured: a request made of a 2,048-token
# passage and a 128-token question, on Qwen2.5-1.5B. The time to its first
# token is one prefill, of all 2,176 tokens, or, with the passage's keys and
# values already cached, of the 128 question tokens. The cache is copied
# before timing, since an engine shares cached blocks instead of copying them.
blog = tok(open("blog_train.txt").read()).input_ids
passage, question = blog[:2048], blog[150_000:150_128]
sync = torch.mps.synchronize if DEVICE == "mps" else torch.cuda.synchronize if DEVICE == "cuda" else (lambda: None)
ids = lambda t: torch.tensor([t], device=DEVICE)

@torch.no_grad()
def first_token(prefix_cache=None):
    if prefix_cache is None:
        cache, new = DynamicCache(), passage + question
    else:
        cache, new = copy.deepcopy(prefix_cache), question
    sync(); t0 = time.perf_counter()
    h = model.model(ids(new), past_key_values=cache, use_cache=True).last_hidden_state
    logits = model.lm_head(h[:, -1])
    sync()
    return time.perf_counter() - t0, logits

with torch.no_grad():
    prefix = DynamicCache()
    model.model(ids(passage), past_key_values=prefix, use_cache=True)
first_token(); first_token(prefix)                                    # warm up
full = sorted(first_token()[0] for _ in range(7))[3]
cached = sorted(first_token(prefix)[0] for _ in range(7))[3]
diff = (first_token()[1] - first_token(prefix)[1]).abs().max().item()
print(f"time to first token, 2,176-token prompt: {full * 1e3:.0f} ms computed whole, "
      f"{cached * 1e3:.0f} ms with the 2,048-token prefix cached ({full / cached:.1f}x faster)")
print(f"largest difference in the first token's logits: {diff:.3f}; same first token: "
      f"{bool(first_token()[1].argmax() == first_token(prefix)[1].argmax())}")
