from lm import tok
from detokenizer import Detokenizer

# A character split across tokens, decoded one token at a time and
# incrementally.
text = "Con vẹt 🦜 nói: mô hình sinh văn bản từng token một 🫠"
ids = tok(text).input_ids
naive = [tok.decode([i]) for i in ids]
print(f"{len(ids)} tokens; decoded one by one:")
print("  " + " | ".join(naive))

d = Detokenizer()
pieces = [d.push(i) for i in ids]
print("incremental:")
print("  " + " | ".join(pieces))
print(f"broken pieces one by one: {sum('�' in p for p in naive)}, incremental: "
      f"{sum('�' in p for p in pieces)}; joined text equal to the original: {''.join(pieces) == text}")