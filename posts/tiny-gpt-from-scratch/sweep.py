import json
import subprocess
import sys

# How the post ran train.py: as a script, one configuration per run, each
# printing its dev-slice bits per byte at the checkpoints.
def train(name, **kw):
    out = subprocess.run([sys.executable, "train.py", name, json.dumps(kw)], capture_output=True, text=True)
    print(out.stdout.strip() if out.returncode == 0 else out.stderr[-2000:], flush=True)

if __name__ == "__main__":
    # The first round of choices: characters or BPE tokens, and how much dropout. One seed each.
    CK = [250, 500, 750, 1000, 1500, 2000, 2500, 3000]
    train("char d0.1 s0", kind="char", seed=0, steps=3000, checkpoints=CK)
    train("bpe  d0.1 s0", kind="bpe", seed=0, steps=3000, checkpoints=CK)
    train("char d0.2 s0", kind="char", seed=0, steps=3000, dropout=0.2, checkpoints=CK)
    train("bpe  d0.2 s0", kind="bpe", seed=0, steps=3000, dropout=0.2, checkpoints=CK)
    CKC, CKB = [1000, 2000, 3000, 4000, 5000, 6000], [250, 500, 750, 1000, 1250, 1500, 2000]
    train("char d0.2 s0", kind="char", seed=0, steps=6000, dropout=0.2, checkpoints=CKC)
    train("char d0.3 s0", kind="char", seed=0, steps=6000, dropout=0.3, checkpoints=CKC)
    train("bpe  d0.3 s0", kind="bpe", seed=0, steps=2000, dropout=0.3, checkpoints=CKB)
    train("bpe  d0.4 s0", kind="bpe", seed=0, steps=2000, dropout=0.4, checkpoints=CKB)
