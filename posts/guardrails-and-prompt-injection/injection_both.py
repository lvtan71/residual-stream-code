import os
import subprocess
import sys

# The post ran injection.py once per model size.
for size in ("1.5B", "3B"):
    subprocess.run([sys.executable, "injection.py"], env={**os.environ, "SIZE": size}, check=True)
