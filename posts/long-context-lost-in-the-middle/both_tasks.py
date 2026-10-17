import subprocess
import sys

# position.py on both tasks: the key-value lookup, then the needle in a haystack.
for task in ("kv", "needle"):
    subprocess.run([sys.executable, "position.py", task], check=True)
