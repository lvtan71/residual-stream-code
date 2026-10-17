import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# The vectors: every sentence of WikiText-2's training set (over 40
# characters), embedded with the all-MiniLM-L6-v2 of "Embeddings and
# Contrastive Learning" (384 dimensions, normalized), on the CPU. Queries: 1,000 sentences of WikiText-2's test
# set, which none of the vectors contain.
import re
W = snapshot_download("Salesforce/wikitext", repo_type="dataset",
                      revision="b08601e04326c79dfdd32d625aee71d232d685c3") + "/wikitext-2-raw-v1/"
def paragraphs(split):
    lines = pq.read_table(W + f"{split}-00000-of-00001.parquet")["text"].to_pylist()
    paras = [l.strip() for l in lines if l.strip() and not l.strip().startswith("=")]
    return [x for p in paras for x in re.split(r"(?<=[.!?]) (?=[A-Z])", p) if len(x) > 40]
docs, queries = paragraphs("train"), paragraphs("test")[:1000]
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", device="cpu")
X = enc.encode(docs, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype("float32")
Q = enc.encode(queries, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype("float32")
np.save("docs.npy", X); np.save("queries.npy", Q)
print(f"{len(docs):,} sentences embedded, {X.shape[1]} dimensions; {len(queries):,} queries")