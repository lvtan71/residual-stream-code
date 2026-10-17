import subprocess
import sys

# The post's final runs: final.py with three seeds and the cosine schedule.
# Seed 0 also saves the checkpoints gpt_250.pt ... gpt_4000.pt that
# per_post.py, repeats.py and generate.py read.
for seed in (0, 1, 2):
    subprocess.run([sys.executable, "final.py", str(seed), "cosine"], check=True)
