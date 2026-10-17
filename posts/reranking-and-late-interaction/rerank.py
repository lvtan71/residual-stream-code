import time
import numpy as np
from sentence_transformers import CrossEncoder
from retrieval import BM25, dense_passages, en, rrf, unique, vi   # data from "Semantic and Hybrid Search", BM25, dense, RRF

# Two-stage retrieval on the benchmark from "Semantic and Hybrid Search". Stage 1, cheap, over every
# section: the best ranker from "Semantic and Hybrid Search" for the language (English: BM25 fused with
# dense passages; Vietnamese: BM25). Stage 2, expensive, over its top 50
# only: a cross-encoder reads query and section together and scores the
# pair (English: ms-marco-MiniLM-L6-v2; multilingual:
# mmarco-mMiniLMv2-L12-H384-v1). Sections it ranks are cut to its input
# limit, 512 tokens; one run instead scores 80-word passages and keeps each
# section's best.
K = 50
REVISION = {"cross-encoder/ms-marco-MiniLM-L6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": "1427fd652930e4ba29e8149678df786c240d8825"}
def two_stage(title, corpus, queries, stage1, ce_name, passages=False):
    ce = CrossEncoder(ce_name, device="cpu", revision=REVISION[ce_name])
    docs = [s["body"] for s in corpus]
    before, after, seconds = [], [], []
    for q, target in queries:
        s1 = stage1(q)
        top = np.argsort(-s1)[:K]
        before.append(int((s1 >= s1[target]).sum()))
        t0 = time.perf_counter()
        if passages:                                                     # score each 80-word passage, keep the best
            pairs, owner = [], []
            for j, i in enumerate(top):
                w = docs[i].split()
                for p in range(0, len(w), 80):
                    pairs.append((q, " ".join(w[p:p + 80]))); owner.append(j)
            sc = ce.predict(pairs, batch_size=64)
            s2 = np.full(K, -np.inf); np.maximum.at(s2, np.array(owner), sc)
        else:
            s2 = ce.predict([(q, docs[i]) for i in top], batch_size=K)
        seconds.append(time.perf_counter() - t0)
        if target in top:
            after.append(int((s2 >= s2[list(top).index(target)]).sum()))
        else:
            after.append(before[-1])                                     # stage 1 missed it; rerank can't help
    b, a = np.array(before), np.array(after)
    d = 1 / a - 1 / b
    rng = np.random.default_rng(0)
    lo, hi = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)], [2.5, 97.5])
    print(f"{title} ({len(queries)} queries), cross-encoder {ce_name.split('/')[1]}"
          f"{', best 80-word passage' if passages else ''}:")
    print(f"  stage 1 alone:    recall@1 {np.mean(b == 1):4.0%}  MRR {np.mean(1 / b):.3f}  (right section in its top {K}: {np.mean(b <= K):.0%})")
    print(f"  + re-rank top {K}: recall@1 {np.mean(a == 1):4.0%}  MRR {np.mean(1 / a):.3f}  "
          f"difference {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"  re-ranking time: {np.median(seconds) * 1e3:.0f} ms per query on the CPU ({K} pairs)")

en_docs, vi_docs = [s["body"] for s in en], [s["body"] for s in vi]
en_q = [(s["heading"], i) for i, s in enumerate(en) if s in unique(en)]
vi_q = [(s["heading"], i) for i, s in enumerate(vi) if s in unique(vi)]
en_stage1 = rrf(BM25(en_docs).scores, dense_passages("MiniLM (English)", en_docs))
vi_stage1 = BM25(vi_docs).scores
two_stage("English", en, en_q, en_stage1, "cross-encoder/ms-marco-MiniLM-L6-v2")
two_stage("English", en, en_q, en_stage1, "cross-encoder/ms-marco-MiniLM-L6-v2", passages=True)
two_stage("English", en, en_q, en_stage1, "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
two_stage("Vietnamese", vi, vi_q, vi_stage1, "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
two_stage("Vietnamese", vi, vi_q, vi_stage1, "cross-encoder/ms-marco-MiniLM-L6-v2")