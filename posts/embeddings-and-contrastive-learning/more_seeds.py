import subprocess
import sys

# word2vec.py again with seeds 1 and 2: stable.py and contrastive.py compare
# the three seeds. Each run saves word2vec1.pt or word2vec2.pt.
for seed in (1, 2):
    subprocess.run([sys.executable, "word2vec.py", str(seed)], check=True)
