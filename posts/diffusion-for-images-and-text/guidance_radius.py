import math
import torch
# Where guided samples go: for each sample, its distance from the center
# of the ring (the true modes sit at radius 2) and its spread around the
# mode it was asked for (the data's own spread is 0.05 per coordinate).
src = open("diff2d.py").read().split("nets = [train(s)")[0]
ns = {}; exec(src.replace("print(", "(lambda *a, **k: None)("), ns)
nets = []
for sd in torch.load("nets.pt"):
    m = ns["Eps"](); m.load_state_dict(sd); nets.append(m)
asked = torch.arange(8).repeat_interleave(250)
print("DDIM 50 steps, 250 samples per mode, all three seeds pooled")
print(f"{'w':>3s} {'median radius':>14s} {'spread around the mode asked for':>33s}")
for w in (1.0, 3.0, 7.0):
    torch.manual_seed(200)
    xs = torch.cat([ns["ddim"](m, 2000, 50, c=asked, w=w) for m in nets])
    target = ns["centers"][asked.repeat(3)]
    print(f"{w:3.0f} {xs.norm(dim=1).median():14.2f} {(xs - target).std(0).mean():33.3f}")