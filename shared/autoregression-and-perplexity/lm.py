import math
import sys
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# An LSTM language model of the blog, over characters or over byte-level BPE
# tokens, trained and scored the same way: predict each next unit from the
# ones before it. The BPE tokenizer is trained on the training text alone
# (shown in "Byte-Pair Encoding"), with every one of the 256 bytes in its vocabulary, so
# any held-out text can be written in its tokens.
torch.set_num_threads(4)
train_text, val_text = open("blog_train.txt").read(), open("blog_val.txt").read()
bpe = Tokenizer(models.BPE())
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.train_from_iterator([train_text], trainers.BpeTrainer(
    vocab_size=2048, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))

def units(text, kind):
    return list(text) if kind == "char" else bpe.encode(text).ids

class LSTMLM(torch.nn.Module):
    def __init__(self, V, hidden=256):
        super().__init__()
        self.embed = torch.nn.Embedding(V, 64)
        self.rnn = torch.nn.LSTM(64, hidden, batch_first=True)
        self.out = torch.nn.Linear(hidden, V)
    def forward(self, x, state=None):
        h, state = self.rnn(self.embed(x), state)
        return self.out(h), state

def build(kind):
    """Units as tensors. Characters never seen in training share id 0; BPE
    tokens use the tokenizer's own ids, all of which exist from the start."""
    tr, va = units(train_text, kind), units(val_text, kind)
    if kind == "char":
        vocab = sorted(set(tr))
        index = {u: i + 1 for i, u in enumerate(vocab)}
        to_t = lambda us: torch.tensor([index.get(u, 0) for u in us])
        return vocab, to_t(tr), to_t(va)
    return list(range(bpe.get_vocab_size() - 1)), torch.tensor(tr), torch.tensor(va)

@torch.no_grad()
def total_nats(model, data, T=128):
    """Sum of -log p over every unit of `data`, carrying the state along."""
    total, state = 0.0, None
    for s in range(0, len(data) - 1, T):
        x, y = data[s:s + T], data[s + 1:s + T + 1]
        logits, state = model(x[None][:, :len(y)], state)
        total += torch.nn.functional.cross_entropy(logits[0], y, reduction="sum").item()
    return total

def train(kind, seed, steps=3000, B=64, T=128, checkpoints=()):
    """Returns the model and, at each checkpoint, validation bits per byte."""
    vocab, tr, va = build(kind)
    n_bytes, seen = len(val_text.encode("utf-8")), {}
    torch.manual_seed(seed)
    model, g = LSTMLM(len(vocab) + 1), torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    for step in range(1, steps + 1):
        i = torch.randint(0, len(tr) - T - 1, (B,), generator=g)
        x = torch.stack([tr[j:j + T] for j in i]); y = torch.stack([tr[j + 1:j + T + 1] for j in i])
        logits, _ = model(x)                                  # teacher forcing: every position at once
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in checkpoints:
            seen[step] = total_nats(model, va) / math.log(2) / n_bytes
    return model, vocab, va, seen

if __name__ == "__main__":
    kind, seed = sys.argv[1], int(sys.argv[2])
    model, vocab, va, seen = train(kind, seed, checkpoints=(250, 500, 1000, 2000, 3000))
    tr = build(kind)[1]
    unseen = (~torch.isin(va, tr)).float().mean().item()      # held-out units absent from training
    nats = total_nats(model, va)
    per_unit = nats / (len(va) - 1)
    passes = 3000 * 64 * 128 / len(tr)
    params = sum(p.numel() for p in model.parameters())
    print(f"{kind:4s} seed {seed}: {len(vocab) + 1:,} units, {params:,} parameters, "
          f"{len(val_text.encode()) / len(va):.2f} bytes per unit, {unseen:.1%} of held-out units unseen "
          f"in training; at 3,000 steps ({passes:.0f} passes) perplexity {math.exp(per_unit):.1f} per unit")
    print("      validation bits per byte at 250 / 500 / 1,000 / 2,000 / 3,000 steps: "
          + " / ".join(f"{v:.3f}" for v in seen.values()))
    if kind == "char" and seed == 0:
        torch.save({"model": model.state_dict(), "vocab": vocab}, "char_lstm.pt")
