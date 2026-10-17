import torch

# Learn to embed a passage so that two pieces of the same blog section land
# close together: SimCLR's idea (two "views" of one thing are a positive pair,
# everything else in the batch is a negative), with random crops as the views.
from sections import W2V, load, train_secs, test_secs, crop, to_ids, VIEW

class Encoder(torch.nn.Module):
    """Average the (frozen) word2vec vectors, then a learned linear map."""
    def __init__(self, init):
        super().__init__()
        self.emb = torch.nn.Embedding.from_pretrained(init.clone(), freeze=True)
        self.proj = torch.nn.Linear(init.shape[1], init.shape[1])
        torch.nn.init.eye_(self.proj.weight); torch.nn.init.zeros_(self.proj.bias)
    def forward(self, windows):                                # a list of word windows
        pooled = torch.stack([self.emb(to_ids(w)).mean(0) for w in windows])
        return torch.nn.functional.normalize(self.proj(pooled), dim=-1)

@torch.no_grad()
def recall_at_1(encode, secs=test_secs):
    """The first 40 words of each section should find that section's last 40
    words among all the sections' last 40 words."""
    a = encode([s[:VIEW] for s in secs])
    b = encode([s[-VIEW:] for s in secs])
    return ((a @ b.T).argmax(1) == torch.arange(len(a))).float().mean().item()

mean_of = lambda E: lambda ws: torch.nn.functional.normalize(
    torch.stack([E[to_ids(w)].mean(0) for w in ws]), dim=-1)
torch.manual_seed(0)
random_emb = torch.nn.functional.normalize(torch.randn_like(W2V), dim=1)
print(f"{len(train_secs)} training sections, {len(test_secs)} held-out sections")
print(f"before training, the encoder is exactly the word2vec average: recall@1 {recall_at_1(Encoder(W2V)):.3f}")
print(f"random word vectors, averaged:   recall@1 {recall_at_1(mean_of(random_emb)):.3f}")
w2v = []
for seed in (0, 1, 2):
    vocab_s, E_s = load(seed)
    idx = {w: i for i, w in enumerate(vocab_s)}
    w2v.append(recall_at_1(lambda ws: torch.nn.functional.normalize(
        torch.stack([E_s[to_ids(w, idx)].mean(0) for w in ws]), dim=-1)))
print("word2vec vectors, averaged:      recall@1 " + ", ".join(f"{v:.3f}" for v in w2v)
      + " (3 word2vec seeds)")

for B in (4, 16, 64):
    scores, fit = [], []
    for seed in range(3):
        g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
        enc = Encoder(W2V)
        opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
        for step in range(1500):
            pick = torch.randint(0, len(train_secs), (B,), generator=g)
            x1 = [crop(train_secs[i], g) for i in pick]
            x2 = [crop(train_secs[i], g) for i in pick]
            logits = enc(x1) @ enc(x2).T / 0.1                  # InfoNCE, temperature 0.1
            same = pick[:, None] == pick[None, :]               # repeats in a batch aren't negatives
            logits = logits.masked_fill(same & ~torch.eye(B, dtype=torch.bool), float("-inf"))
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(B))
            opt.zero_grad(); loss.backward(); opt.step()
        scores.append(recall_at_1(enc))
        fit.append(recall_at_1(enc, train_secs[:len(test_secs)]))   # same number of training sections
    print(f"contrastive, batch {B:3d} ({B - 1:2d} negatives): recall@1 "
          + ", ".join(f"{s:.3f}" for s in scores) + " (3 seeds); on training sections "
          + ", ".join(f"{s:.3f}" for s in fit))