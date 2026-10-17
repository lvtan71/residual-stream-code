import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# WikiText-2 (raw), from the Hugging Face Hub at a pinned revision: the
# training split becomes the text to train on and choose with, the test
# split the held-out text. About 7 MB of parquet.
REPO, REVISION = "Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3"

def text(split):
    path = hf_hub_download(REPO, f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset",
                           revision=REVISION)
    return "".join(pq.read_table(path)["text"].to_pylist())

tr, te = text("train"), text("test")
open("wikitext2_train.txt", "w").write(tr); open("wikitext2_test.txt", "w").write(te)
print(len(tr), len(te), "passes in 4000 steps of 32x256:", round(4000 * 32 * 256 / (0.9 * len(tr)), 1))
