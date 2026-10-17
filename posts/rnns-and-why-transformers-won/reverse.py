import subprocess
import sys

# Each length, with and without attention, two seeds, at seq2seq.py's default 2,000 steps.
for L in (5, 10, 20, 40):
    for attn in ("none", "attention"):
        for seed in (0, 1):
            subprocess.run([sys.executable, "seq2seq.py", str(L), attn, str(seed)], check=True)
