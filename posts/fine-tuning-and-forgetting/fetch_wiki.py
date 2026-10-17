import os
import shutil
from huggingface_hub import hf_hub_download

# The post's run read one shard each of English and Vietnamese Wikipedia
# (the November 2023 dump) from wiki/. This fetches the same two files,
# about 0.7 GB, at a pinned revision.
os.makedirs("wiki", exist_ok=True)
for lang, name in (("en", "20231101.en/train-00001-of-00041.parquet"), ("vi", "20231101.vi/train-00002-of-00004.parquet")):
    path = hf_hub_download("wikimedia/wikipedia", name, repo_type="dataset",
                           revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa")
    if not os.path.exists(f"wiki/{lang}.parquet"):   # an earlier run may have fetched it
        shutil.copyfile(path, f"wiki/{lang}.parquet")    # copyfile: the cache file is read-only