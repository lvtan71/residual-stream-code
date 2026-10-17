import glob
import os
import re

# The blog's own English posts, as plain text: frontmatter, imports and
# component tags removed; prose, math and code kept. The series posts written
# for this part are held out, so a model can't have memorized its own text.
POSTS = "content/posts"
HELD_OUT = {"kv-cache-explained", "bpe-tokenizer-explained"}      # validation posts

def clean(text):
    text = re.sub(r"\A---.*?---\n", "", text, flags=re.S)         # frontmatter
    text = re.sub(r"^import .*$", "", text, flags=re.M)           # MDX imports
    text = re.sub(r"<[A-Z][^>]*?/>", "", text, flags=re.S)        # self-closing components
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

def load():
    train, val = [], []
    for path in sorted(glob.glob(f"{POSTS}/*.mdx")):
        slug = os.path.basename(path)[:-4]
        (val if slug in HELD_OUT else train).append(clean(open(path).read()))
    return "".join(train), "".join(val)

if __name__ == "__main__":
    train, val = load()
    chars = sorted(set(train))
    print(f"train {len(train):,} characters, validation {len(val):,}; {len(chars)} distinct characters")
    print(f"validation characters never seen in training: {sum(c not in set(train) for c in val)}")
