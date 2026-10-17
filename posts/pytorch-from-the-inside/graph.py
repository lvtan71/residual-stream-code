import torch

x = torch.tensor([[1.0, 0.0]])
W1 = torch.tensor([[0.5, -0.4], [0.3, 0.8]], requires_grad=True)
w2 = torch.tensor([[0.7], [-0.6]], requires_grad=True)
z = torch.relu(x @ W1) @ w2

# The graph PyTorch recorded while computing z, walked from the output back.
def walk(fn, depth=0):
    if fn is None:
        return
    print("  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        walk(nxt, depth + 1)
walk(z.grad_fn)

# The graph is rebuilt on every forward pass, so ordinary Python control flow
# decides its shape: here, how many times a layer is applied.
def count(fn, seen=None):
    seen = set() if seen is None else seen
    if fn is None or fn in seen:
        return 0
    seen.add(fn)
    return 1 + sum(count(nxt, seen) for nxt, _ in fn.next_functions)

W = torch.eye(2, requires_grad=True)
for repeats in (1, 3):
    h = x
    for _ in range(repeats):
        h = torch.relu(h @ W)
    print(f"layer applied {repeats}x: {count(h.grad_fn)} nodes in the graph")

# Under no_grad nothing is recorded: no graph, and no saved activations.
with torch.no_grad():
    print(f"under no_grad: grad_fn = {torch.relu(x @ W1).grad_fn}")