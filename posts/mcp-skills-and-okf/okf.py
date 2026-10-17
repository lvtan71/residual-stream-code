import json
import os
import yaml

# An OKF bundle (Open Knowledge Format v0.2) for a few of this blog's
# concepts: a directory of Markdown files, each with YAML frontmatter whose
# only required field is `type`, plus an index. Generated from the posts'
# own front matter, so it records where it came from and that no person has
# checked it yet. The concepts are ideas, not assets, so they carry no
# `resource`; the post each was derived from goes in `sources`.
POSTS = "content/posts"
SITE = "https://residual-stream.lvtan71.workers.dev"
concepts = {"kv-cache": "kv-cache-explained", "lora": "parameter-efficient-fine-tuning-lora",
            "speculative-decoding": "speculative-decoding", "hnsw": "vector-search-and-ann-indexes"}
os.makedirs("bundle/concepts", exist_ok=True)
index = ["---", "okf_version: \"0.2\"", "---", "", "# Residual Stream concepts", ""]
for name, slug in concepts.items():
    path = f"{POSTS}/{slug}.mdx"
    if not os.path.exists(path):
        continue                                              # a post not yet written (lora, here)
    front = yaml.safe_load(open(path).read().split("---")[1])
    title, desc = front["title"], " ".join(front["description"].split())
    doc = f"""---
type: Concept
title: {json.dumps(title, ensure_ascii=False)}
description: {json.dumps(desc[:200] + "...", ensure_ascii=False)}
tags: [llm, inference]
sources:
  - resource: {SITE}/posts/{slug}/
    title: {json.dumps(title, ensure_ascii=False)}
generated:
  by: process:make_okf
  at: "2026-10-01"
status: draft
---

# Summary

{desc}

# See also

[the full post]({SITE}/posts/{slug}/)
"""
    open(f"bundle/concepts/{name}.md", "w").write(doc)
    index.append(f"- [{title}](concepts/{name}.md)")
open("bundle/index.md", "w").write("\n".join(index) + "\n")
print(open("bundle/index.md").read())
print(open("bundle/concepts/kv-cache.md").read()[:900])
