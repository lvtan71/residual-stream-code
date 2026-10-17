import json
import subprocess
import sys

# Runs train.py once per configuration, as the post did, and prints each
# run's bits per byte at the checkpoints.
def train(name, **kw):
    out = subprocess.run([sys.executable, "train.py", name, json.dumps(kw)], capture_output=True, text=True)
    print(out.stdout.strip() if out.returncode == 0 else out.stderr[-2000:], flush=True)

# 8 experts on the blog's characters, top-1 with each balancing method and
# top-2 with the balancing loss, two seeds each, 4,000 steps on the dev slice.
BASE = dict(kind="char", steps=4000, dropout=0.2, checkpoints=[1000, 2000, 3000, 4000])
for seed in (0, 1):
    train(f"8 experts, top-1, balancing loss   s{seed}", **BASE, seed=seed, experts=8, top_k=1, balance="aux")
    train(f"8 experts, top-1, no balancing     s{seed}", **BASE, seed=seed, experts=8, top_k=1, balance="none")
    train(f"8 experts, top-1, bias balancing   s{seed}", **BASE, seed=seed, experts=8, top_k=1, balance="bias")
    train(f"8 experts, top-2, balancing loss   s{seed}", **BASE, seed=seed, experts=8, top_k=2, balance="aux")
