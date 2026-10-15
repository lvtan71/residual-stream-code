import math
import torch
import torch.nn.functional as F
from gpt import GPT, Config
from train import dev_split, final_split, tokenizer_for

# How much of the held-out text repeats something, and the model's loss on
# it. A character counts as "copied" if it lies inside a stretch of SPAN or
# more characters that also appears word for word in the training text, and
# as "repeated" if that stretch appeared earlier in the held-out text itself,
# more than 256 characters back, beyond the model's window.
torch.set_num_threads(4)

def copied_mask(text, train, span):
    covered = torch.zeros(len(text), dtype=torch.bool)
    for i in range(len(text) - span + 1):
        if text[i:i + span] in train:
            covered[i:i + span] = True
    return covered

def repeated_mask(text, span, T=256):
    covered, first_seen = torch.zeros(len(text), dtype=torch.bool), {}
    for i in range(len(text) - span + 1):
        s = text[i:i + span]
        if s in first_seen and first_seen[s] < i - T:
            covered[i:i + span] = True
        first_seen.setdefault(s, i)
    return covered

@torch.no_grad()
def bits_each(model, data, T):
    """Bits for each character from the second on, scored as in train.py."""
    out = torch.zeros(len(data) - 1)
    s = 0
    while True:
        x = data[s:s + T + 1]
        logits = model(x[None, :-1])[0]
        first = 0 if s == 0 else T // 2
        out[s + first:s + len(x) - 1] = F.cross_entropy(logits[first:], x[1 + first:], reduction="none") / math.log(2)
        if s + T + 1 >= len(data):
            break
        s += T // 2
    return out

if __name__ == "__main__":
    ckpt = torch.load("gpt_4000.pt", weights_only=False)
    model = GPT(Config(**ckpt["config"])); model.load_state_dict(ckpt["model"]); model.eval()
    train, text = final_split["fit"], final_split["score"]
    _, encode, _ = tokenizer_for("char", train)
    bits = bits_each(model, encode(text), model.c.T)             # bits per character, characters 1 onward
    print(f"the model on the held-out posts: {bits.mean():.3f} bits per character")
    for span in (32, 16):
        copied = copied_mask(text, train, span)[1:]
        rep = repeated_mask(text, span)[1:] & ~copied
        rest = ~copied & ~rep
        print(f"stretches of {span}+ characters: copied {copied.float().mean():5.1%} at {bits[copied].mean():.2f} bits, "
              f"repeated {rep.float().mean():5.1%} at {bits[rep].mean():.2f}, the rest {rest.float().mean():5.1%} "
              f"at {bits[rest].mean():.2f}")
    dev_copied = copied_mask(dev_split["score"], dev_split["fit"], 32).float().mean()
    print(f"for comparison, the dev slice: {dev_copied:.1%} copied from the rest of the training text (32+)")
