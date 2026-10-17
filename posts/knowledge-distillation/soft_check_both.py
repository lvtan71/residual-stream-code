import subprocess
import sys

# soft_check.py at the post's learning rate and at ten times it.
for lr in ("1e-5", "1e-4"):
    subprocess.run([sys.executable, "soft_check.py", lr], check=True)
