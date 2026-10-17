import torch

# Count the bytes a training step actually keeps, for a small MLP trained
# with AdamW, the way mixed precision usually runs in PyTorch: FP32 weights,
# BF16 math inside autocast.
model = torch.nn.Sequential(torch.nn.Linear(1024, 4096), torch.nn.ReLU(), torch.nn.Linear(4096, 1024))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(64, 1024)
saved = []                                              # what autograd keeps for backward
weights = {p.shape for p in model.parameters() if p.dim() == 2}
with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(x).float().pow(2).mean()
loss.backward()
opt.step()

size = lambda ts: sum(t.numel() * t.element_size() for t in ts)
dtype = lambda ts: str(next(iter(ts)).dtype).split(".")[1]
n = sum(p.numel() for p in model.parameters())
state = [t for s in opt.state.values() for t in s.values() if t.dim() > 0]
parts = {"weights": list(model.parameters()),
         "gradients": [p.grad for p in model.parameters()],
         "Adam m and v": state}
for name, ts in parts.items():
    print(f"{name:13s} {size(ts) / n:4.1f} bytes/param  ({dtype(ts)})")
print(f"{'total':13s} {sum(size(ts) for ts in parts.values()) / n:4.1f} bytes/param for {n:,} parameters")
copies = [t for t in saved if t.dtype == torch.bfloat16 and t.shape in weights]
print(f"during the step: {len(copies)} BF16 weight copies saved for backward, "
      f"{size(copies) / n:.1f} more bytes/param")