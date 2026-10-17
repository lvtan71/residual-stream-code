import subprocess
import sys

# train_position.py once per task, position scheme and seed: 28 runs, one line each.
for task in ("lag", "seen"):
    for kind in ("none", "sinusoid", "learned", "rope", "rope:1000000", "alibi:0.5", "alibi:0.0625"):
        for seed in (0, 1):
            subprocess.run([sys.executable, "train_position.py", kind, str(seed), task], check=True)
