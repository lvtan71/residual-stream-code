import glob
import math
import os
import re
from collections import Counter
from mcp.server.mcpserver import MCPServer

# An MCP server for this blog: two tools and one resource, over stdio. Any
# MCP host (an IDE, a desktop assistant, an agent) can launch it and use
# them, without code written for that host.
POSTS = os.environ.get("POSTS", "content/posts")

def load():
    out = {}
    for path in glob.glob(f"{POSTS}/*.mdx"):
        raw = open(path).read()
        title = re.search(r'^title: "(.*)"$', raw, re.M)
        body = re.sub(r"```.*?```|\$\$.*?\$\$|^import .*?$", " ", raw.split("---", 2)[-1], flags=re.S | re.M)
        out[os.path.basename(path)[:-4]] = (title.group(1) if title else "", " ".join(body.split()))
    return out
posts = load()
words = lambda t: re.findall(r"\w+", t.lower())
counts = {s: Counter(words(b)) for s, (_, b) in posts.items()}
df = Counter(w for c in counts.values() for w in c)
avg = sum(sum(c.values()) for c in counts.values()) / len(counts)

server = MCPServer("residual-stream-blog", instructions="Search and read the Residual Stream blog's posts.")

@server.tool()
def search_posts(query: str, k: int = 3) -> list[dict]:
    """Find the blog posts most relevant to a query (BM25 over the full text).
    Returns each post's slug, title and score."""
    def bm25(slug):
        c, n = counts[slug], sum(counts[slug].values())
        return sum(math.log((len(posts) - df[w] + 0.5) / (df[w] + 0.5) + 1) * c[w] * 2.5
                   / (c[w] + 1.5 * (0.25 + 0.75 * n / avg)) for w in set(words(query)) if w in df)
    best = sorted(posts, key=bm25, reverse=True)[:k]
    return [{"slug": s, "title": posts[s][0], "score": round(bm25(s), 2)} for s in best]

@server.tool()
def read_post(slug: str, max_words: int = 400) -> str:
    """Read the start of one post, by slug."""
    if slug not in posts:
        raise ValueError(f"no post {slug!r}")
    return " ".join(posts[slug][1].split()[:max_words])

@server.resource("post://{slug}")
def post_resource(slug: str) -> str:
    """A post's full text, for a host to attach as context."""
    return posts[slug][1]

if __name__ == "__main__":
    server.run()                                  # stdio by default
