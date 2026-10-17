import time
import torch
from lm import LSTMLM

# Generation is the factorization run forward: predict a distribution for the
# next character, pick one, append it, repeat.
torch.set_num_threads(4)
saved = torch.load("char_lstm.pt")
vocab = saved["vocab"]
index = {c: i + 1 for i, c in enumerate(vocab)}
model = LSTMLM(len(vocab) + 1)
model.load_state_dict(saved["model"]); model.eval()

@torch.no_grad()
def generate(prompt, n, greedy, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits, state = model(torch.tensor([[index[c] for c in prompt]]))
    out = []
    for _ in range(n):
        p = torch.softmax(logits[0, -1], -1)
        nxt = p.argmax() if greedy else torch.multinomial(p, 1, generator=g)[0]
        out.append(vocab[nxt - 1] if nxt > 0 else "?")
        logits, state = model(nxt.view(1, 1), state)       # feed its own output back in
    return "".join(out)

prompt = "The KV cache "
print(f"greedy:  {prompt}{generate(prompt, 200, greedy=True)!r}")
print(f"sampled: {prompt}{generate(prompt, 200, greedy=False)!r}")

# Three ways through 512 characters, median of 7 runs each: scoring known
# text in one call (teacher forcing), feeding the same known text one call
# per character, and generating (each step waits for the character just chosen).
text = open("blog_val.txt").read()[:513]
x = torch.tensor([[index.get(c, 0) for c in text[:-1]]])

def median_ms(fn, runs=7):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    return 1e3 * sorted(times)[runs // 2]

@torch.no_grad()
def one_call_per_char():
    state = None
    for t in range(x.shape[1]):
        _, state = model(x[:, t:t + 1], state)

with torch.no_grad():
    forced = median_ms(lambda: model(x))
stepwise, generated = median_ms(one_call_per_char), median_ms(lambda: generate("T", 512, greedy=True))
print(f"512 characters: teacher forcing {forced:.1f} ms; the same known characters one call at a time "
      f"{stepwise:.1f} ms; generating {generated:.1f} ms")