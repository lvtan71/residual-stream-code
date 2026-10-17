import glob
import os
import re

# The blog as clean text, shared by every script in this post: no
# frontmatter, code, display math, components or tables; links reduced to
# their text and emphasis marks removed; "## " headings kept. Posts that
# discuss this post's experiments are left out.
EXCLUDE = {"building-rag-chunking-hyde-and-caching", "function-calling-to-agent-loop",
           "agentic-graph-and-vectorless-rag", "recursive-language-models"}

def clean(raw):
    text = raw.split("---", 2)[-1]
    text = re.sub(r"```.*?```|\$\$.*?\$\$", "\n\n", text, flags=re.S)
    text = re.sub(r"^import .*$|<[^>]+>|^\|.*$", "\n", text, flags=re.M)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)                  # links -> their text
    text = re.sub(r"\*\*|__|(?<![\w*])[*_](?=\w)|(?<=\w)[*_](?![\w*])", "", text)  # bold, italics
    text = re.sub(r"^#{3,} ", "## ", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def posts():
    out = {}
    for path in sorted(glob.glob("content/posts/*.mdx")):
        slug = os.path.basename(path)[:-4]
        if slug not in EXCLUDE:
            raw = open(path).read()
            out[slug] = (re.search(r'^title: "?(.*?)"?$', raw, re.M).group(1), clean(raw))
    return out
