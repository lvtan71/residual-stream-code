import math
import torch

def info_nce(a, b, tau):
    """Row i of a should match row i of b and no other row: cross-entropy over
    the similarity matrix, with the diagonal as the correct 'class' ("Cross-Entropy and the Softmax Gradient")."""
    a, b = torch.nn.functional.normalize(a, dim=1), torch.nn.functional.normalize(b, dim=1)
    logits = a @ b.T / tau
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(a))), a @ b.T

# Three pairs of 2-D embeddings: each b is a slightly rotated copy of its a.
a = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.2]])
b = torch.tensor([[0.9, 0.3], [0.2, 1.0], [-1.0, -0.1]])
loss, sim = info_nce(a, b, 1.0)
print("cosine similarities (rows: a, columns: b):")
for row in sim:
    print("   " + "  ".join(f"{v:+.3f}" for v in row))
for tau in (1.0, 0.5, 0.1, 0.05):
    print(f"temperature {tau:<4}: loss {info_nce(a, b, tau)[0]:.4f}")

# Random, unrelated embeddings: the loss is about log(batch size), a uniform
# guess among B ("Cross-Entropy and the Softmax Gradient"), and the best possible loss is 0. The bound:
# mutual information >= log B - loss. (At a low temperature, random
# similarities spread out and the loss goes above log B.)
torch.manual_seed(0)
for B in (4, 64, 1024):
    x, y = torch.randn(B, 64), torch.randn(B, 64)
    print(f"batch {B:5d}: random embeddings give loss {info_nce(x, y, 1.0)[0]:.2f} at temperature 1, "
          f"{info_nce(x, y, 0.1)[0]:.2f} at 0.1; log B = {math.log(B):.2f}")