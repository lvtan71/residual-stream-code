import glob
import os
import re

# The benchmark: every section of this blog's posts (a snapshot taken for
# this post). The query is a section's heading, the one relevant document
# is that section's text. Headings every post shares (takeaways, references
# and so on) are left out, as are sections under 200 characters.
SKIP = {"table of contents", "takeaways", "references", "where this leads", "mục lục", "những điểm chính",
        "tài liệu tham khảo", "tiếp theo", "bài học rút ra", "tóm tắt", "kết luận", "conclusion",
        "những điều rút ra", "hướng đi tiếp theo"}

def sections(lang):
    out = []
    for path in sorted(glob.glob(f"corpus/{lang}/*.mdx")):
        text = open(path).read().split("---", 2)[-1]
        text = re.sub(r"```.*?```", " ", text, flags=re.S)              # code blocks
        text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)                # display math
        text = re.sub(r"\]\([^)]*\)", "]", text)                           # link targets (URLs, slugs)
        text = re.sub(r"<[^>]+>|^import .*$", " ", text, flags=re.M)        # components, imports
        parts = re.split(r"^#{2,3} (.+)$", text, flags=re.M)
        for heading, body in zip(parts[1::2], parts[2::2]):
            h = re.sub(r"[*_`]", "", heading).strip()
            if h.lower().rstrip(":") in SKIP or len(body.strip()) < 200:
                continue
            out.append({"post": os.path.basename(path)[:-4], "heading": h, "body": " ".join(body.split())})
    return out
