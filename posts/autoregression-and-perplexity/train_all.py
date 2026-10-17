import subprocess
import sys

# The post trained each unit (characters, BPE tokens) with two seeds, one
# `python lm.py <kind> <seed>` each. Seed 0 of the character model also saves
# char_lstm.pt, which generate.py reads.
for kind in ("char", "bpe"):
    for seed in (0, 1):
        subprocess.run([sys.executable, "lm.py", kind, str(seed)], check=True)
