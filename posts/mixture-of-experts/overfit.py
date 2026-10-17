import json
import subprocess
import sys

# Runs train.py once per configuration, as the post did, and prints each
# run's bits per byte at the checkpoints.
def train(name, **kw):
    out = subprocess.run([sys.executable, "train.py", name, json.dumps(kw)], capture_output=True, text=True)
    print(out.stdout.strip() if out.returncode == 0 else out.stderr[-2000:], flush=True)

# The dense model and the top-1 MoE again, seed 0, on the CPU, also scoring
# 60,000 characters of the training text at each checkpoint (track_fit).
BASE = dict(kind="char", steps=4000, dropout=0.2, checkpoints=[1000, 2000, 3000, 4000], seed=0, device="cpu",
            track_fit=True)
train("dense                              s0", **BASE)
train("8 experts, top-1, balancing loss   s0", **BASE, experts=8, top_k=1, balance="aux")
