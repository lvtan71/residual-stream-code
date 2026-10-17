import subprocess
import sys

# 12 more key-value trials at 8,000 tokens (120 pairs) with new random draws
# (the salt "-b"), and 12 at 16,000 tokens (240 pairs).
subprocess.run([sys.executable, "position.py", "kv", "120,240", "12", "-b"], check=True)
