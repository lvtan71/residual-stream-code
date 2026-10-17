import subprocess
import sys

# 40 digits, the model without attention at its better learning rate (3e-3), trained 6,000 steps.
for seed in (0, 1):
    subprocess.run([sys.executable, "seq2seq.py", "40", "none", str(seed), "6000"], check=True)
