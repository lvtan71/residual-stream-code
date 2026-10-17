import subprocess
import sys

# The same three cells trained four times as long, 12,000 steps, seed 0.
for cell in ("RNN", "GRU", "LSTM"):
    subprocess.run([sys.executable, "charlm.py", cell, "0", "12000"], check=True)
