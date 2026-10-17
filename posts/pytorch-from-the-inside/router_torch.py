import numpy as np
import torch
from router_data import split
from router4 import four_way
from softmax_basics import softmax
from optim import Adam

((Xtr, ytr), (Xva, yva), _), _ = four_way(*split(0))
X, y = torch.tensor(Xtr, dtype=torch.float64), torch.tensor(ytr)

# The four-way router as a PyTorch module, trained the way "Initialization and Optimizers" trained it.
model = torch.nn.Linear(X.shape[1], 4, dtype=torch.float64)
torch.nn.init.zeros_(model.weight); torch.nn.init.zeros_(model.bias)
opt = torch.optim.AdamW([{"params": [model.weight], "weight_decay": 0.3},
                         {"params": [model.bias], "weight_decay": 0.0}],   # no decay on biases
                        lr=0.01)

# The same run with "Initialization and Optimizers"'s hand-written AdamW, on the same mini-batches.
W, b, ours = np.zeros((Xtr.shape[1], 4)), np.zeros(4), Adam(0.01, decay=0.3)

rng = np.random.default_rng(0)
for step in range(1, 3001):
    i = rng.choice(len(ytr), 16, replace=False)
    loss = torch.nn.functional.cross_entropy(model(X[i]), y[i])   # takes logits ("Cross-Entropy and the Softmax Gradient")
    opt.zero_grad()                                               # gradients accumulate otherwise
    loss.backward()
    opt.step()
    G = (softmax(Xtr[i] @ W + b) - np.eye(4)[ytr[i]]) / 16
    ours.step(W, Xtr[i].T @ G, "W"); ours.step(b, G.sum(0), "b")
    if step in (1, 10, 3000):
        gap = np.abs(model.weight.detach().numpy().T - W).max()
        print(f"step {step:4d}: loss {loss.item():.4f}, largest gap between PyTorch and optim.py weights {gap:.1e}")

with torch.no_grad():
    acc = (model(torch.tensor(Xva)).argmax(1).numpy() == yva).mean()
print(f"validation accuracy {acc:.3f}")