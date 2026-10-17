import subprocess
import sys

# The post ran tts.py with two seeds. The problems and the greedy answers are
# the same for both, so seed 1 prints only the sampled rows.
for seed in (0, 1):
    subprocess.run([sys.executable, "tts.py", str(seed)], check=True)
