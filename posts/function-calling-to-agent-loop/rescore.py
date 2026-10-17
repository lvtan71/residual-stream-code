import glob
import json
import os
import re

# Rescoring the logged answers. agent.py counted an answer as citing the
# right post if the post's name appeared anywhere after "Source:", and
# anywhere at all when there was no "Source:" line; a list of several
# posts also passed. Here a citation counts only if the answer has a
# "Source:" line naming exactly one of the blog's posts, and that post is
# the right one. Answers are also sorted by what their source line holds.
SLUGS = {os.path.basename(p)[:-4] for p in glob.glob("content/posts/*.mdx")}
def cited(answer):
    if "Source:" not in answer:
        return "no source line", None
    line = answer.split("Source:")[-1]
    named = {s for s in SLUGS if s in line}
    if len(named) == 1:
        return "one real post", named.pop()
    if len(named) > 1:
        return "several posts", None
    return "placeholder or invented", None

for model in ("1.5B", "3B"):
    rows = [json.loads(l) for l in open(f"agent_log_{model}.jsonl")]
    print(f"Qwen2.5-{model}-Instruct")
    print(f"  {'':32s} {'found':>6s} {'cites right':>12s}   source line: none / one real post / several / placeholder or invented")
    for method in dict.fromkeys(r["method"] for r in rows):
        rs = [r for r in rows if r["method"] == method]
        kinds = [cited(r["answer"]) for r in rs]
        right = sum(slug == r["post"] for (_, slug), r in zip(kinds, rs))
        count = lambda k: sum(kk == k for kk, _ in kinds)
        print(f"  {method:32s} {sum(r['found'] for r in rs):>6d} {right:>12d}   "
              f"{count('no source line'):>4d} / {count('one real post'):>2d} / {count('several posts'):>2d} / "
              f"{count('placeholder or invented'):>2d}")
