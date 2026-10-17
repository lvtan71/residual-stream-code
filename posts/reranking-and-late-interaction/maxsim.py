import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Late interaction (ColBERT's MaxSim), worked on a 3-word query (4 word pieces). Every
# token of the query and of each document gets its own vector; each query
# token takes its best match among the document's tokens, and the score is
# the sum of those maxima. The token vectors here are all-MiniLM-L6-v2's,
# which was not trained for this: the arithmetic is ColBERT's, the numbers
# only illustrate it.
path, rev = "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
tok, enc = AutoTokenizer.from_pretrained(path, revision=rev), AutoModel.from_pretrained(path, revision=rev).eval()
def vectors(text):
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        h = enc(**ids).last_hidden_state[0, 1:-1]                     # drop [CLS] and [SEP]
    return tok.convert_ids_to_tokens(ids.input_ids[0, 1:-1]), torch.nn.functional.normalize(h, dim=-1).numpy()

q_tokens, Qv = vectors("paged kv cache")
for doc in ("vllm stores the kv cache in fixed-size pages, like virtual memory",
            "the cache line of a cpu holds sixty-four bytes"):
    d_tokens, Dv = vectors(doc)
    S = Qv @ Dv.T                                                     # every query token vs every document token
    best = S.argmax(1)
    print(f'document: "{doc}"')
    for i, t in enumerate(q_tokens):
        print(f"  {t:7s} best match {d_tokens[best[i]]!r:12s} {S[i, best[i]]:.2f}   (mean over the document {S[i].mean():.2f})")
    print(f"  MaxSim score {S.max(1).sum():.2f}; mean-of-similarities score {S.mean(1).sum():.2f}")