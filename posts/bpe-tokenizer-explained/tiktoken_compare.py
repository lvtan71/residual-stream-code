"""OpenAI's released tokenizers on the post's examples (tiktoken)."""

import tiktoken

ENCODINGS = ("gpt2", "cl100k_base", "o200k_base")


def pieces(enc, text):
    """Each token's text; a token that is not whole UTF-8 shows as its hex bytes."""
    out = []
    for t in enc.encode(text):
        b = enc.decode_single_token_bytes(t)
        try:
            out.append(b.decode("utf-8").replace(" ", "␣"))
        except UnicodeDecodeError:
            out.append("<" + " ".join(f"{x:02X}" for x in b) + ">")
    return out


for name in ENCODINGS:
    print(f"{name:12s} vocabulary {tiktoken.get_encoding(name).n_vocab:,}")

gpt2 = tiktoken.get_encoding("gpt2")
for text in ("accomplishment", " accomplishment", "the lowest widest fox newest lower"):
    p = pieces(gpt2, text)
    print(f"\ngpt2 {text!r}: {len(p)} token{'s' * (len(p) != 1)}  {' | '.join(p)}")

for text in ("12345678", "Xin chào thế giới"):
    print(f"\n{text!r}: {len(text)} characters, {len(text.encode())} bytes")
    for name in ENCODINGS:
        p = pieces(tiktoken.get_encoding(name), text)
        print(f"  {name:12s} {len(p):2d} tokens  {' | '.join(p)}")
