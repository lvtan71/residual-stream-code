import math
import torch
import torch.nn as nn

# A learned simulator, and what happens when you run it on its own output.
# The world: a damped pendulum (angle, angular velocity), stepped 0.05 s at
# a time. A small network learns one step, (state) -> (next state), from
# 20,000 random transitions. Then "imagination": start from real states and
# feed the model its own predictions for H steps, as Dreamer-style agents do
# to plan, and compare with the real trajectory.
wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi          # angles live in [-pi, pi)
def step(x, dt=0.05):
    th, om = x[:, 0], x[:, 1]
    om = om + dt * (-9.8 * torch.sin(th) - 0.1 * om)
    return torch.stack([wrap(th + dt * om), om], 1)
def sample_states(n):
    # the training range covers every state the test trajectories reach,
    # so errors below come from compounding, not from leaving the data
    return torch.stack([(torch.rand(n) * 2 - 1) * math.pi, (torch.rand(n) * 2 - 1) * 10], 1)

def train(seed, n=20000, steps=4000):
    torch.manual_seed(seed)
    x = sample_states(n); y = step(x)
    net = nn.Sequential(nn.Linear(2, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(steps):
        i = torch.randint(0, n, (512,))
        d = torch.stack([wrap(y[i, 0] - x[i, 0]), y[i, 1] - x[i, 1]], 1)
        loss = ((net(x[i]) - d) ** 2).mean()                # predict the change in state
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda s: torch.stack([wrap(s[:, 0] + net(s)[:, 0]), s[:, 1] + net(s)[:, 1]], 1)

horizons = (1, 5, 15, 50, 100)
print("error of imagined angles against the real ones (radians; a random guess would be")
print("off by about 1.8 root-mean-square), 1,000 start states, 3 seeds")
print(f"{'steps imagined':>22s} " + " ".join(f"{h:>8d}" for h in horizons))
errs, meds = torch.zeros(3, len(horizons)), torch.zeros(3, len(horizons))
for seed in range(3):
    model = train(seed)
    with torch.no_grad():
        torch.manual_seed(100)
        # start anywhere with |angle|, |velocity| <= 2.5: some swings, some spins over the top
        real = imag = (torch.rand(1000, 2) * 2 - 1) * 2.5
        h_i = 0
        for h in range(1, max(horizons) + 1):
            real, imag = step(real), model(imag)
            if h in horizons:
                d = wrap(real[:, 0] - imag[:, 0]).abs()
                errs[seed, h_i], meds[seed, h_i] = (d ** 2).mean().sqrt(), d.median(); h_i += 1
for name, v in (("RMS error, seed mean", errs.mean(0)), ("median error, seed mean", meds.mean(0))):
    print(f"{name:>22s} " + " ".join(f"{x:8.4f}" for x in v))
print(f"{'RMS, per seed at 15':>22s} " + " ".join(f"{x:8.4f}" for x in errs[:, 2]))