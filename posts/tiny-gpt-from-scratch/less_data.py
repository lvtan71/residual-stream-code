import subprocess
import sys

# datafrac.py on the first quarter and the first half of the training text, two seeds each.
for frac in (0.25, 0.5):
    for seed in (0, 1):
        subprocess.run([sys.executable, "datafrac.py", str(frac), str(seed)], check=True)
