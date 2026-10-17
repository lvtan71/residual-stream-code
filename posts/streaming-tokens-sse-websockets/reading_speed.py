import json
from lm import tok

# Is generation faster than reading? Tokens per word on the blog's own
# prose (the 20 real continuations from "Decoding"), against a reader's speed: 238
# words a minute for non-fiction, the average across studies (Brysbaert,
# 2019). Generation speed is from measure.py: 100 tokens in about 4.7 s.
texts = [p["human"] for p in json.load(open("prompts.json"))]
tokens = sum(len(tok(t).input_ids) for t in texts)
words = sum(len(t.split()) for t in texts)
per_word = tokens / words
reading = 238 / 60 * per_word                              # tokens a second a reader gets through
generating = 100 / 4.7
print(f"{per_word:.2f} tokens per word; a reader at 238 words a minute takes in {reading:.1f} tokens a second; "
      f"this laptop generates {generating:.0f}, {generating / reading:.1f} times as fast")
for n_words in (50, 300):
    n = n_words * per_word
    print(f"a {n_words}-word answer ({n:.0f} tokens): {n / generating:.1f} s to generate, {n_words / 238 * 60:.0f} s to read; "
          f"without streaming the reader waits {n / generating:.1f} s before the first word, with it about 0.13 s")