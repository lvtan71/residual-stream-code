import json
import subprocess
import sys

# Runs train.py once per configuration, as the post did, and prints each
# run's bits per byte at the checkpoints.
def train(name, **kw):
    out = subprocess.run([sys.executable, "train.py", name, json.dumps(kw)], capture_output=True, text=True)
    print(out.stdout.strip() if out.returncode == 0 else out.stderr[-2000:], flush=True)

# The dense model and the MoE, top-1 and top-2, on WikiText-2 (written by
# wikitext2.py) instead of the blog, two seeds each, 4,000 steps.
BASE = dict(kind="char", steps=4000, dropout=0.2, checkpoints=[1000, 2000, 3000, 4000], corpus="wikitext2")
for seed in (0, 1):
    train(f"dense (WikiText-2)                 s{seed}", **BASE, seed=seed)
    train(f"8 experts, top-1, balancing loss   s{seed}", **BASE, seed=seed, experts=8, top_k=1, balance="aux")
    train(f"8 experts, top-2, balancing loss   s{seed}", **BASE, seed=seed, experts=8, top_k=2, balance="aux")
