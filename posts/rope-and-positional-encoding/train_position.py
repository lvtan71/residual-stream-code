import math
import sys
import torch

# A tiny causal Transformer, trained on sequences of 32 and tested on 32, 64
# and 128, on one of two tasks:
#   lag:  output, at every position, the token 3 positions earlier. It needs
#         relative position, always at the same short distance.
#   seen: say whether the current token appeared anywhere earlier. It needs
#         to look back over every distance, including ones longer than any
#         training sequence.
# Kinds: none, sinusoid, learned, rope (or rope:<base>), alibi:<slope>.
torch.set_num_threads(2)
TASK = sys.argv[3] if len(sys.argv) > 3 else "lag"
V = 16 if TASK == "lag" else 96
D, LAG, TRAIN_T = 64, 3, 32
SCALE = 1.0          # multiplies every attention score; 1 in training

def targets(x):
    """(the positions scored, their correct outputs)"""
    if TASK == "lag":
        return slice(LAG, None), x[:, :-LAG]
    earlier = (x[:, :, None] == x[:, None, :]).tril(-1).any(-1)   # token t appeared before t
    return slice(1, None), earlier[:, 1:].long()

def sinusoids(T, D):
    pos, i = torch.arange(T)[:, None], torch.arange(0, D, 2)
    pe = torch.zeros(T, D)
    pe[:, 0::2] = torch.sin(pos / 10_000 ** (i / D)); pe[:, 1::2] = torch.cos(pos / 10_000 ** (i / D))
    return pe

def rope(x, base=10_000.0):                                   # x: (batch, T, D)
    T, half = x.shape[1], x.shape[2] // 2
    ang = torch.arange(T)[:, None] * base ** (-torch.arange(half) / half)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()], -1).flatten(2)

class Layer(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind, _, arg = kind.partition(":")
        self.arg = float(arg) if arg else 10_000.0
        self.qkv, self.o = torch.nn.Linear(D, 3 * D), torch.nn.Linear(D, D)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(D, 4 * D), torch.nn.GELU(), torch.nn.Linear(4 * D, D))
        self.n1, self.n2 = torch.nn.LayerNorm(D), torch.nn.LayerNorm(D)
    def forward(self, x):
        T = x.shape[1]
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        if self.kind == "rope":
            q, k = rope(q, self.arg), rope(k, self.arg)
        s = SCALE * q @ k.transpose(1, 2) / math.sqrt(D)
        if self.kind == "alibi":                                # a penalty growing with distance
            dist = torch.arange(T)[:, None] - torch.arange(T)[None]
            s = s - self.arg * dist.clamp(min=0)
        s = s.masked_fill(torch.ones(T, T).triu(1).bool(), float("-inf"))
        x = x + self.o(torch.softmax(s, -1) @ v)
        return x + self.mlp(self.n2(x))

class Model(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind.partition(":")[0]
        self.emb, self.out = torch.nn.Embedding(V, D), torch.nn.Linear(D, V if TASK == "lag" else 2)
        self.pos = torch.nn.Embedding(128, D) if kind == "learned" else None
        self.layers = torch.nn.ModuleList([Layer(kind) for _ in range(2)])
    def forward(self, x):
        h = self.emb(x)
        T = x.shape[1]
        if self.kind == "sinusoid":
            h = h + sinusoids(T, D)
        if self.kind == "learned":
            h = h + self.pos(torch.arange(T))                   # rows past 32 are never trained
        for layer in self.layers:
            h = layer(h)
        return self.out(h)

def accuracy(model, T, g, past=0):
    """Accuracy on 256 fresh sequences of length T, counting only
    positions from `past` on."""
    x = torch.randint(0, V, (256, T), generator=g)
    where, y = targets(x)
    with torch.no_grad():
        pred = model(x)[:, where].argmax(-1)
    first = max(0, past - where.start)
    return (pred == y)[:, first:].float().mean().item()

kind, seed = sys.argv[1], int(sys.argv[2])
torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
model = Model(kind)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(2000):
    x = torch.randint(0, V, (64, TRAIN_T), generator=g)
    where, y = targets(x)
    logits = model(x)[:, where]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
model.eval()
test = lambda: torch.Generator().manual_seed(100 + seed)       # the same test sequences each time
g = test()
line = " / ".join(f"{accuracy(model, T, g):.3f}" for T in (32, 64, 128))
past = accuracy(model, 128, test(), past=32)
SCALE = math.log(128) / math.log(32)                               # log-n scaling, at test time only
scaled = accuracy(model, 128, test())
print(f"{TASK:4s} {kind:12s} seed {seed}: length 32 / 64 / 128: {line}; "
      f"at 128, positions past 32: {past:.3f}; scores x log128/log32: {scaled:.3f}")