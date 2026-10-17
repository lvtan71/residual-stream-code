import subprocess
import sys

# The three recurrent cells, three seeds each, at charlm.py's default 3,000 steps.
for cell in ("RNN", "GRU", "LSTM"):
    for seed in (0, 1, 2):
        subprocess.run([sys.executable, "charlm.py", cell, str(seed)], check=True)
