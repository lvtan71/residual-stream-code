import math
import sys
import time
import torch

# A character-level language model of this blog: read 128 characters, predict
# each next one. Three recurrent cells, the same size, the same training.
train_text, val_text = open("blog_train.txt").read(), open("blog_val.txt").read()
chars = sorted(set(train_text))
index = {c: i + 1 for i, c in enumerate(chars)}           # 0 = a character never seen in training
encode = lambda s: torch.tensor([index.get(c, 0) for c in s])
train, val = encode(train_text), encode(val_text)
V = len(chars) + 1

class CharRNN(torch.nn.Module):
    def __init__(self, cell, hidden=256):
        super().__init__()
        self.embed = torch.nn.Embedding(V, 64)
        self.rnn = {"RNN": torch.nn.RNN, "GRU": torch.nn.GRU, "LSTM": torch.nn.LSTM}[cell](
            64, hidden, batch_first=True)
        self.out = torch.nn.Linear(hidden, V)
    def forward(self, x, state=None):
        h, state = self.rnn(self.embed(x), state)
        return self.out(h), state

def batch(data, rng, B=64, T=128):
    i = torch.randint(0, len(data) - T - 1, (B,), generator=rng)
    x = torch.stack([data[j:j + T] for j in i])
    return x, torch.stack([data[j + 1:j + T + 1] for j in i])

@torch.no_grad()
def evaluate(model, data, T=128):
    """Average loss per character over the whole text, carrying the state along."""
    total, state = 0.0, None
    for s in range(0, len(data) - 1 - T, T):
        logits, state = model(data[s:s + T][None], state)
        total += torch.nn.functional.cross_entropy(logits[0], data[s + 1:s + T + 1], reduction="sum").item()
    return total / ((len(data) - 1 - T) // T * T)

cell, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3000
torch.manual_seed(seed)
model, rng = CharRNN(cell), torch.Generator().manual_seed(seed)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
t0 = time.perf_counter()
for step in range(1, steps + 1):
    x, y = batch(train, rng)
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # "Initialization and Optimizers"
    opt.step()
val_loss = evaluate(model, val)
train_loss = evaluate(model, train[:len(val)])                 # the same amount of training text
params = sum(p.numel() for p in model.parameters())
print(f"{cell:4s} seed {seed}, {steps:,} steps: {params:,} parameters, {time.perf_counter() - t0:.0f} s; "
      f"train {train_loss / math.log(2):.3f}, validation {val_loss / math.log(2):.3f} bits/char")
