import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from common import posts

# Rescoring the scorecard's saved answers with stricter rules (no model
# calls). A refusal is an answer that begins with the refusal sentence;
# a citation is right if any passage the answer cites comes from the right
# post, counted only for answers that aren't refusals. Also: how many
# answer sentences can be found in any chunk at all, and whether each
# cache hit matched its own original question.
REFUSAL = "The blog doesn't cover this."
rows, items = json.load(open("answers.json")), json.load(open("gen.json"))
src = open("notebook.py").read()
CHUNK_WORDS = 100
exec(src[src.index("def chunks_of"):src.index("def build_index")])          # the assistant's own chunker
chunks = [c for slug, (t, x) in posts().items() for c in chunks_of(slug, t, x)]
norm = lambda t: " ".join(t.split())
findable = sum(any(norm(it["sentence"]) in norm(c["text"]) for c in chunks) for it in items)
refused = [r["answer"].strip().startswith(REFUSAL) for r in rows]
answered = [r for r, f in zip(rows, refused) if not f]
print(f"answer sentence findable in some chunk: {findable}/{len(items)}; among the 4 passages: "
      f"{sum(r['retrieved'] for r in rows)}/{len(rows)}")
print(f"refused (answer begins with the refusal): {sum(refused)}; of them, with the sentence retrieved: "
      f"{sum(r['retrieved'] for r, f in zip(rows, refused) if f)}; refusals that still cite a passage: "
      f"{sum(bool(re.search(r'\[\d+\]', r['answer'])) for r, f in zip(rows, refused) if f)}")
print(f"answers: {len(answered)}; citing the right post among them: {sum(r['cites_post'] for r in answered)}; "
      f"citing more than one passage: {sum(len(set(re.findall(r'\[(\d+)\]', r['answer']))) > 1 for r in answered)}")
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41", device="cpu")
Q = m.encode([it["question"] for it in items], normalize_embeddings=True)
R = m.encode([it["reworded"][0] for it in items[:30]], normalize_embeddings=True)
S = R @ Q.T
hits = [i for i in range(30) if S[i].max() >= 0.90]
print(f"cache: {len(hits)} of 30 rewordings hit at 0.90; matched their own original: "
      f"{sum(S[i].argmax() == i for i in hits)}; cached answers that were refusals: "
      f"{sum(refused[i] for i in hits)}")