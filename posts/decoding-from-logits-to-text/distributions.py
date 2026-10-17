import json
import math
import torch
from lm import model, tok, encode
from sampling import top_k, top_p, min_p

# Two real next-token distributions: one where the model is nearly sure,
# one where many continuations are reasonable. For each: the most likely
# tokens, the entropy at three temperatures, and how many tokens each
# filter keeps.
def entropy_nats(logits):
    p = logits.softmax(-1)
    return -(p * p.clamp_min(1e-30).log()).sum().item()

kept = lambda l: int((l > float("-inf")).sum())
for prompt in ("The KV cache stores the keys and", "My favorite thing about this blog is"):
    with torch.no_grad():
        logits = model(encode(prompt)[None]).logits[0, -1].float()
    p = logits.softmax(-1)
    top = p.topk(6)
    print(f"{prompt!r} ->")
    print("   " + ", ".join(f"{tok.decode(int(i))!r} {v:.3f}" for v, i in zip(top.values, top.indices)))
    print("   entropy at T = 0.5 / 1 / 1.5: " + " / ".join(f"{entropy_nats(logits / t):.2f}" for t in (0.5, 1, 1.5))
          + f" nats;  top token's probability at T = 0.1: {(logits / 0.1).softmax(-1).max():.4f}")
    print(f"   tokens kept at T = 1: top-k 40 keeps {kept(top_k(logits.clone(), 40))}, "
          f"top-p 0.9 keeps {kept(top_p(logits.clone(), 0.9))}, min-p 0.1 keeps {kept(min_p(logits.clone(), 0.1))}")
    for t in (1.5, 2.0):
        print(f"   at T = {t} (temperature applied first): top-p 0.9 keeps {kept(top_p(logits / t, 0.9))}, "
              f"min-p 0.1 keeps {kept(min_p(logits / t, 0.1))}; with the filters first, top-p's set stays "
              f"{kept(top_p(logits.clone(), 0.9))} and min-p's {kept(min_p(logits.clone(), 0.1))}")

# The open prompt's distribution, for the widget: the 200 most likely tokens
# one by one, and every other token as a histogram of logits, in bins 0.25
# wide, so the widget can still apply temperature and filters to the tail.
top = logits.topk(200)
rest = logits.sort(descending=True).values[200:]
bins = torch.floor(rest / 0.25) * 0.25 + 0.125                    # each bin's centre
centres, counts = bins.unique(return_counts=True)
print(f"the widget's 200 tokens cover {p[top.indices].sum():.1%} of the probability at T = 1; "
      f"the other {len(rest):,} tokens go in {len(centres)} bins")
json.dump({"prompt": prompt, "tokens": [tok.decode(int(i)) for i in top.indices],
           "logits": [round(v, 3) for v in top.values.tolist()],
           "tail": [[round(c, 3), int(n)] for c, n in zip(centres.tolist()[::-1], counts.tolist()[::-1])]},
          open("widget_dist.json", "w"), ensure_ascii=False)