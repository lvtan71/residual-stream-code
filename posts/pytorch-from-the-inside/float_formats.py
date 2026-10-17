import torch

bits = {torch.float32: (8, 23), torch.float16: (5, 10), torch.bfloat16: (8, 7)}
print(f"{'':9s} {'exponent':>8s} {'mantissa':>8s} {'largest':>10s} {'smallest normal':>16s} "
      f"{'smallest subnormal':>19s} {'step at 1.0':>12s}")
for dt, (e, m) in bits.items():
    f = torch.finfo(dt)                                   # a subnormal is the normal minimum's last bits
    print(f"{str(dt).split('.')[1]:9s} {e:8d} {m:8d} {f.max:10.3g} {f.tiny:16.3g} "
          f"{f.tiny * f.eps:19.3g} {f.eps:12.3g}")

show = lambda v, dt: torch.tensor(v, dtype=torch.float32).to(dt).item()
print()
for v in (1e-8, 70_000.0, 1.001):
    print(f"{v:>9g} stored as  float16: {show(v, torch.float16):<12.6g} bfloat16: {show(v, torch.bfloat16):.6g}")

# Adding many small updates to one number, as an optimizer does to a weight.
for dt in (torch.float32, torch.bfloat16):
    w = torch.tensor(1.0, dtype=dt)
    for _ in range(10_000):
        w = w + torch.tensor(1e-4, dtype=dt)
    print(f"1 + 10,000 steps of 0.0001 in {str(dt).split('.')[1]:8s} = {w.item():.4f} (should be 2.0000)")

# FP16's narrow range: small gradients vanish unless the loss is scaled up first.
g = torch.tensor([1e-8, 2e-7, 1e-4])
fmt = lambda t: "[" + ", ".join(f"{v:.3g}" for v in t.tolist()) + "]"
print(f"gradients {fmt(g)} in float16: {fmt(g.half().float())}")
print(f"scaled by 1024, stored, unscaled: {fmt((g * 1024).half().float() / 1024)}")