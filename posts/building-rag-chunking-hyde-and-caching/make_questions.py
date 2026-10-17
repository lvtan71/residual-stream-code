import json
import random
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The test set for this post, made from the blog itself. 150 sentences of
# prose are drawn at random from the posts (no code, math, tables or
# headings; 15 to 45 words; at most 3 per post). For each, Qwen2.5-3B-
# Instruct writes:
#   a question the sentence answers, in different words;
#   two rewordings of that question (repeats, for the semantic cache);
#   a short hypothetical answer written without seeing the blog (for HyDE).
# The sentence is the gold answer: retrieval hits if a retrieved chunk
# contains all of it.
from common import posts
from huggingface_hub import snapshot_download
rng = random.Random(0)
cands = []
for slug, (title, text) in posts().items():
    sents = []
    for para in re.split(r"\n\s*\n|\n(?=\s*(?:[-*]|\d+\.) )", text):    # paragraphs and list items
        if para.lstrip().startswith(("#", "-", "*")) or re.match(r"\s*\d+\. ", para):
            continue
        para = " ".join(para.split())
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", para):
            if 15 <= len(s.split()) <= 45 and "$" not in s and "`" not in s and not s.startswith(("-", "*", "1.")):
                sents.append(s)
    cands += [(slug, s) for s in rng.sample(sents, min(3, len(sents)))]
items = rng.sample(cands, 150)

PATH = snapshot_download("Qwen/Qwen2.5-3B-Instruct",
                         revision="aa8e72537993ba99e69dfaafa59ed015b17504d1")
tok = AutoTokenizer.from_pretrained(PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
lm = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16).to(DEVICE).eval()
@torch.no_grad()
def ask(prompt, max_new=120):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to(DEVICE)
    out = lm.generate(ids, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

out = []
for slug, sent in items:
    q = ask("Write one question that the following sentence answers. Use your own words: don't copy its "
            f"distinctive phrases. Reply with the question only.\n\nSentence: {sent}")
    reworded = [ask(f"Reword this question {how}. Keep its meaning. Reply with the question only.\n\n{q}")
                for how in ("as a user typing quickly might", "more formally")]
    hyde = ask(f"Answer this question in two or three sentences, as a technical blog post would.\n\n{q}", 100)
    out.append({"post": slug, "sentence": sent, "question": q, "reworded": reworded, "hyde": hyde})
    print(json.dumps(out[-1])[:200], flush=True)
json.dump(out, open("gen_new.json", "w"), indent=1)   # the post's set is gen.json, linked; this run doesn't overwrite it