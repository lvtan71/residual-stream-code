import sys
import torch
from gpt import GPT, Config

# Sample from the saved model at several checkpoints: feed the last T
# characters, take the distribution for the next one, sample, repeat. Then
# check memorization: the longest stretch of each sample that appears word
# for word in the training posts.
train = open("blog_train.txt").read()

def longest_copied(text):
    best, i = "", 0
    for i in range(len(text)):
        j = i + len(best) + 1
        while j <= len(text) and text[i:j] in train:
            best, j = text[i:j], j + 1
    return best

@torch.no_grad()
def sample(path, prompt, n=300, temperature=0.8, seed=0, greedy=False):
    ckpt = torch.load(path, weights_only=False)
    c = Config(**ckpt["config"])
    model = GPT(c); model.load_state_dict(ckpt["model"]); model.eval()
    vocab = ckpt["vocab"]
    index = {ch: i + 1 for i, ch in enumerate(vocab)}
    g = torch.Generator().manual_seed(seed)
    ids = [index.get(ch, 0) for ch in prompt]
    for _ in range(n):
        logits = model(torch.tensor([ids[-c.T:]]))[0, -1] / temperature
        ids.append(logits.argmax().item() if greedy else
                   torch.multinomial(torch.softmax(logits, -1), 1, generator=g).item())
    return "".join(vocab[i - 1] if i > 0 else "?" for i in ids)

prompt = sys.argv[1] if len(sys.argv) > 1 else "The KV cache"
for step in (250, 1000, 4000):
    text = sample(f"gpt_{step}.pt", prompt)
    copied = longest_copied(text[len(prompt):])
    print(f"--- after {step:,} steps (temperature 0.8)\n{text}\n"
          f"[longest stretch copied from the training posts: {len(copied)} characters: {copied!r}]\n")

# For comparison: text a person wrote, never trained on. Three 300-character
# pieces of the held-out posts, and their longest stretch found in training.
held_out = open("blog_val.txt").read()
for start in (10_000, 25_000, 40_000):
    copied = longest_copied(held_out[start:start + 300])
    print(f"held-out text from character {start:,}: longest stretch also in the training posts "
          f"{len(copied)} characters: {copied!r}")

# Ten more samples from the same prompt, and greedy decoding (always the
# most likely next character) from a prompt that looks like the start of
# one of the blog's code blocks.
lengths = [len(longest_copied(sample("gpt_4000.pt", prompt, seed=s)[len(prompt):])) for s in range(1, 11)]
print(f"\nten more samples at 4,000 steps: longest copied stretch {min(lengths)} to {max(lengths)} characters")
code = "```python\nimport torch\n"
text = sample("gpt_4000.pt", code, n=200, greedy=True)
copied = longest_copied(text[len(code):])
print(f"greedy, from a code prompt:\n{text}\n[longest stretch copied: {len(copied)} characters: {copied!r}]")