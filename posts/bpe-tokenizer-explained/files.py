"""Save and load a tokenizer as GPT-2 does: vocab.json and merges.txt."""

import json


def bytes_to_unicode():
    """GPT-2's table from the 256 byte values to 256 printable characters."""
    keep = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1)))  # printable: map to themselves
    table, n = {}, 0
    for b in range(256):
        if b in keep:
            table[b] = chr(b)
        else:
            table[b] = chr(256 + n)  # the rest move past Latin-1, in order
            n += 1
    return table


B2U = bytes_to_unicode()
U2B = {c: b for b, c in B2U.items()}


def to_text(token):
    return "".join(B2U[b] for b in token)


def to_token(text):
    return bytes(U2B[c] for c in text)


def save(vocab, merges, prefix):
    with open(f"{prefix}_vocab.json", "w") as f:
        json.dump({to_text(t): i for i, t in vocab.items()}, f, ensure_ascii=False)
    with open(f"{prefix}_merges.txt", "w") as f:
        f.writelines(f"{to_text(a)} {to_text(b)}\n" for a, b in merges)


def load(prefix, specials=("<|endoftext|>",)):
    with open(f"{prefix}_vocab.json") as f:
        vocab = {i: (t.encode() if t in specials else to_token(t))
                 for t, i in json.load(f).items()}
    with open(f"{prefix}_merges.txt") as f:
        merges = [tuple(to_token(x) for x in line.rstrip("\n").split(" ")) for line in f]
    return vocab, merges


if __name__ == "__main__":
    import os
    import tempfile

    from bpe import CORPUS, SPECIAL, train

    print(f"bytes kept as themselves: {sum(B2U[b] == chr(b) for b in range(256))}, "
          f"moved: {sum(B2U[b] != chr(b) for b in range(256))}")
    print("space ->", B2U[32], " newline ->", B2U[10], " byte 0 ->", B2U[0])
    vocab, merges = train(CORPUS, 265, specials=[SPECIAL])
    prefix = os.path.join(tempfile.mkdtemp(), "toy")
    save(vocab, merges, prefix)
    print("\ntoy_vocab.json, beyond the bytes:",
          {t: i for t, i in json.load(open(f"{prefix}_vocab.json")).items() if i >= 256})
    print("toy_merges.txt:")
    print(open(f"{prefix}_merges.txt").read(), end="")
    print("\nload(save(...)) gives back the same tokenizer:",
          load(prefix) == (vocab, merges))
