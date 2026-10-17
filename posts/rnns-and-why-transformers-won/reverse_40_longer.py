import subprocess
import sys

# 40 digits, both models, 3,000 steps at the original learning rate (3e-3).
for attn in ("none", "attention"):
    for seed in (0, 1):
        subprocess.run([sys.executable, "seq2seq.py", "40", attn, str(seed), "3000"], check=True)
