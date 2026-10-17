import time
import torch
from lm import model, encode, DEVICE
from sampling import choose
sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(DEVICE, lambda: None)   # the post ran on "mps"

# What a decoding step costs: the model's forward pass for one new token
# (with the KV cache), against the sampling policy that picks the token.
prompt = encode("The KV cache stores the keys and values of every earlier token, so")
g = torch.Generator(device=prompt.device).manual_seed(0)
with torch.no_grad():
    out = model(prompt[None], use_cache=True)
    logits, cache = out.logits[0, -1], out.past_key_values
    def step():
        model(torch.tensor([[11]], device=prompt.device), past_key_values=cache, use_cache=True)
    def timed(fn, n=30):
        for _ in range(3):
            fn()
        sync(); t0 = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        return (time.perf_counter() - t0) / n * 1e3
    print(f"forward pass for one token: {timed(step):.1f} ms")
    for name, policy in (("greedy", dict(temperature=0)), ("top-p 0.9", dict(p=0.9)),
                         ("T 0.7 + top-k 40 + top-p 0.9 + min-p 0.05 + repetition 1.1",
                          dict(temperature=0.7, k=40, p=0.9, mp=0.05, rep=1.1))):
        print(f"choosing the token, {name}: {timed(lambda: choose(logits, list(range(50)), g, **policy)):.2f} ms")