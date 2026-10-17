import json
import subprocess
import sys

# Runs train.py once per configuration, as the post did, and prints each
# run's dev-slice bits per byte at the checkpoints.
def train(name, **kw):
    out = subprocess.run([sys.executable, "train.py", name, json.dumps(kw)], capture_output=True, text=True)
    print(out.stdout.strip() if out.returncode == 0 else out.stderr[-2000:], flush=True)

# The tiny GPT's SwiGLU MLP and RMSNorm against a GELU MLP at 4D and against
# LayerNorm, one swap at a time, two seeds each, on the dev slice.
BASE = dict(kind="char", steps=4000, dropout=0.2, checkpoints=[1000, 2000, 3000, 4000])
for seed in (0, 1):
    train(f"SwiGLU + RMSNorm (baseline) s{seed}", **BASE, seed=seed)
    train(f"GELU MLP, 4D wide         s{seed}", **BASE, seed=seed, mlp="gelu")
    train(f"LayerNorm instead of RMS  s{seed}", **BASE, seed=seed, norm="layer")
