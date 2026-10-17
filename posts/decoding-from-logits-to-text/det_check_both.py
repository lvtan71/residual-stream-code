# Runs det_check.py in 16-bit and then in 32-bit floats, as the post did.
import subprocess
import sys

for bits in ("16", "32"):
    subprocess.run([sys.executable, "det_check.py", bits], check=True)
