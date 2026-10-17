import numpy as np
import torch
from tiny_autograd import Tensor

# "Backpropagation, by Hand"'s worked example: one XOR input through the 2-2-1 network.
x, y = np.array([[1.0, 0.0]]), np.array([[1.0]])
W1, b1 = np.array([[0.5, -0.4], [0.3, 0.8]]), np.array([0.0, 0.1])
w2, b2 = np.array([[0.7], [-0.6]]), np.array([0.1])

ours = [Tensor(a) for a in (W1, b1, w2, b2)]
loss = ((Tensor(x) @ ours[0] + ours[1]).relu() @ ours[2] + ours[3]).bce_with_logits(y)
loss.backward()

theirs = [torch.tensor(a, requires_grad=True) for a in (W1, b1, w2, b2)]
h = torch.relu(torch.tensor(x) @ theirs[0] + theirs[1])
ref = torch.nn.functional.binary_cross_entropy_with_logits(h @ theirs[2] + theirs[3], torch.tensor(y))
ref.backward()

print(f"loss: ours {loss.data:.4f}, PyTorch {ref.item():.4f}")
for name, a, b in zip(["W1", "b1", "w2", "b2"], ours, theirs):
    print(f"dL/d{name}: {np.round(a.grad, 4).ravel().tolist()}   largest gap to PyTorch "
          f"{np.abs(a.grad - b.grad.numpy()).max():.1e}")

# Beyond the worked example: random inputs, a weight used twice, a (1, n) bias.
worst = 0.0
for seed in range(200):
    rng = np.random.default_rng(seed)
    X, Y = rng.normal(size=(8, 5)), rng.integers(0, 2, (8, 1)).astype(float)
    W, b = 0.5 * rng.normal(size=(5, 5)), rng.normal(size=(1, 5))
    wo = 0.5 * rng.normal(size=(5, 1))
    ours = [Tensor(a) for a in (W, b, wo)]
    h = ((Tensor(X) @ ours[0] + ours[1]).relu() @ ours[0]).relu()       # W used twice
    (h @ ours[2]).bce_with_logits(Y).backward()
    theirs = [torch.tensor(a, requires_grad=True) for a in (W, b, wo)]
    h = torch.relu(torch.relu(torch.tensor(X) @ theirs[0] + theirs[1]) @ theirs[0])
    torch.nn.functional.binary_cross_entropy_with_logits(h @ theirs[2], torch.tensor(Y)).backward()
    worst = max(worst, max(np.abs(a.grad - t.grad.numpy()).max() for a, t in zip(ours, theirs)))
print(f"200 random cases, shared weight and (1, 5) bias: largest gap to PyTorch {worst:.1e}")