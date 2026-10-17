import re
import pandas as pd
from huggingface_hub import snapshot_download

# Data for replay and for a second test of forgetting. Replay draws from raw
# English Wikipedia (one shard of the November 2023 dump), not WikiText's
# training split, so the replayed text doesn't share WikiText's formatting
# ("@-@", "= = Heading = =") with the WikiText test it's scored on; articles
# whose titles appear in WikiText's test set are dropped. The second test is
# Vietnamese Wikipedia, which nothing in training touches (articles over
# 3,000 characters, since much of it is machine-made one-line stubs).
WIKI = snapshot_download("Salesforce/wikitext", repo_type="dataset",
                         revision="b08601e04326c79dfdd32d625aee71d232d685c3") + "/wikitext-2-raw-v1/"
test = "".join(pd.read_parquet(WIKI + "test-00000-of-00001.parquet")["text"])[:120_000]
titles = {re.sub(r" @(.)@ ", r"\1", t.strip()).replace("( ", "(").replace(" )", ")")   # undo WikiText's spacing
          for t in re.findall(r"^ = ([^=].*?) = $", test, flags=re.M)}
en = pd.read_parquet("wiki/en.parquet", columns=["title", "text"])
dropped = int(en["title"].isin(titles).sum())
en = en[~en["title"].isin(titles)].sample(frac=1.0, random_state=0)
replay, n = [], 0
for t in en["text"]:
    replay.append(t); n += len(t)
    if n > 4_000_000:
        break
open("wiki_replay.txt", "w").write("\n\n".join(replay))
vi = pd.read_parquet("wiki/vi.parquet", columns=["text"])
vi = vi[vi["text"].str.len() > 3000].sample(frac=1.0, random_state=0)      # skip the many one-line stubs
out, n = [], 0
for t in vi["text"]:
    out.append(t); n += len(t)
    if n > 120_000:
        break
open("vi_test.txt", "w").write("\n\n".join(out)[:120_000])
print(f"WikiText test articles: {len(titles)}, of which {dropped} are in this shard and dropped; replay text {sum(map(len, replay)):,} chars from {len(replay)} articles")