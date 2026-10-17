import sys
import torch

# Reverse a sequence of random digits: an encoder reads it, a decoder writes it
# backwards. Without attention, the decoder sees only the encoder's last state;
# with attention, it can look back at every encoder state (Bahdanau et al.).
BOS, V, H = 10, 11, 128

class Seq2Seq(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.embed = torch.nn.Embedding(V, 32)
        self.encoder = torch.nn.GRU(32, H, batch_first=True)
        self.decoder = torch.nn.GRUCell(32 + (H if attention else 0), H)
        self.W_dec, self.W_enc = torch.nn.Linear(H, H), torch.nn.Linear(H, H)
        self.v = torch.nn.Linear(H, 1, bias=False)
        self.out = torch.nn.Linear(H, 10)

    def forward(self, src, tgt_in):
        enc, h = self.encoder(self.embed(src))                 # enc: every step; h: the last one
        h, logits = h[0], []
        for t in range(tgt_in.shape[1]):
            x = self.embed(tgt_in[:, t])
            if self.attention:                                 # additive attention over enc
                score = self.v(torch.tanh(self.W_dec(h)[:, None] + self.W_enc(enc)))[..., 0]
                context = (torch.softmax(score, dim=1)[..., None] * enc).sum(1)
                x = torch.cat([x, context], dim=1)
            h = self.decoder(x, h)
            logits.append(self.out(h))
        return torch.stack(logits, 1)

def data(n, L, g):
    src = torch.randint(0, 10, (n, L), generator=g)
    tgt = src.flip(1)
    return src, torch.cat([torch.full((n, 1), BOS), tgt[:, :-1]], 1), tgt

@torch.no_grad()
def greedy(model, src, L):
    y, out = torch.full((len(src), 1), BOS), []
    for _ in range(L):                                        # feed back its own predictions
        nxt = model(src, y)[:, -1].argmax(-1, keepdim=True)
        y = torch.cat([y, nxt], 1)
    return y[:, 1:]

L, attention, seed = int(sys.argv[1]), sys.argv[2] == "attention", int(sys.argv[3])
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-3
torch.manual_seed(seed)
g = torch.Generator().manual_seed(seed)
model = Seq2Seq(attention)
opt = torch.optim.Adam(model.parameters(), lr=lr)
for step in range(steps):
    src, tgt_in, tgt = data(64, L, g)
    loss = torch.nn.functional.cross_entropy(model(src, tgt_in).reshape(-1, 10), tgt.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
src, _, tgt = data(1000, L, torch.Generator().manual_seed(1000 + seed))   # fresh test sequences
pred = greedy(model, src, L)
print(f"length {L:2d}, {'attention' if attention else 'no attention':12s}, seed {seed}"
      f"{'' if steps == 2000 else f', {steps:,} steps'}{'' if lr == 3e-3 else f', learning rate {lr:g}'}: "
      f"digits right {(pred == tgt).float().mean():.3f}, whole sequences right {(pred == tgt).all(1).float().mean():.3f}")