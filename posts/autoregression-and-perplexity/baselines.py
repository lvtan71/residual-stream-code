import bz2
import lzma
import math
import zlib
from collections import Counter

# Baselines for "bits per byte" on the held-out posts, from knowing nothing
# to general-purpose compressors that are allowed to read the training posts.
train, val = open("blog_train.txt").read(), open("blog_val.txt").read()
tb, vb = train.encode(), val.encode()
n = len(vb)
print(f"raw UTF-8 bytes:                         {8.0:.2f} bits per byte")
V = len(set(train)) + 1
print(f"uniform guess over {V} characters:        {math.log2(V) * len(val) / n:.2f}")
freq = Counter(train)
uni = -sum(math.log2((freq[c] + 1) / (len(train) + V)) for c in val) / n
print(f"unigram character counts:                {uni:.2f}")
for name, f in [("zlib -9", lambda b: zlib.compress(b, 9)), ("bz2 -9", lambda b: bz2.compress(b, 9)),
                ("xz -9e", lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))]:
    alone = 8 * len(f(vb)) / n
    given = 8 * (len(f(tb + vb)) - len(f(tb))) / n       # extra bytes to add the held-out posts
    print(f"{name:8s} on the held-out posts alone {alone:.2f}; given the training posts {given:.2f}")