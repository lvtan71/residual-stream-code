import glob
import os
import re

# The blog's own English posts, as plain text: frontmatter, imports and
# component tags removed; prose, math and code kept. Two whole posts are held
# out for validation, so the model is scored on text it has never seen.
# Run once to freeze a snapshot (2026-09-30), so later posts don't change it.
POSTS = "content/posts"
HELD_OUT = {"kv-cache-explained", "bpe-tokenizer-explained"}      # validation posts

def clean(text):
    text = re.sub(r"\A---.*?---\n", "", text, flags=re.S)         # frontmatter
    text = re.sub(r"^import .*$", "", text, flags=re.M)           # MDX imports
    text = re.sub(r"<[A-Z][^>]*?/>", "", text, flags=re.S)        # self-closing components
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

def load():
    train, val = [], []
    paths = sorted(glob.glob(f"{POSTS}/*.mdx"))
    assert paths, f"no posts under {POSTS}/: see the README for the snapshot"
    for path in paths:
        slug = os.path.basename(path)[:-4]
        (val if slug in HELD_OUT else train).append(clean(open(path).read()))
    return "".join(train), "".join(val)

if __name__ == "__main__":
    if os.path.exists("blog_train.txt"):                      # use the frozen snapshot if it's there
        train, val = open("blog_train.txt").read(), open("blog_val.txt").read()
    else:
        train, val = load()
        open("blog_train.txt", "w").write(train)
        open("blog_val.txt", "w").write(val)
    chars = sorted(set(train))
    print(f"train {len(train):,} characters, validation {len(val):,}; {len(chars)} distinct characters")
    print(f"validation characters never seen in training: {sum(c not in set(train) for c in val)} "
          f"of {len(val):,}")