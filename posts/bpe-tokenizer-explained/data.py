"""Download the corpora this post trains and counts on, at pinned revisions.

TinyStories (Eldan & Li, 2023): GPT-4-written children's stories separated by
<|endoftext|>. OPUS-100 en-vi (Zhang et al., 2020): English-Vietnamese
sentence pairs. Files land in the Hugging Face cache; other scripts call
these functions to find them.
"""

import os

from huggingface_hub import hf_hub_download

TINYSTORIES = ("roneneldan/TinyStories", "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64")
OPUS = ("Helsinki-NLP/opus-100", "805090dc28bf78897da9641cdf08b61287580df9")


def tinystories(split):
    """split is "train" (2.2 GB) or "valid" (22 MB)."""
    repo, rev = TINYSTORIES
    return hf_hub_download(repo, f"TinyStoriesV2-GPT4-{split}.txt", repo_type="dataset",
                           revision=rev)


def opus_en_vi():
    repo, rev = OPUS
    return hf_hub_download(repo, "en-vi/test-00000-of-00001.parquet", repo_type="dataset",
                           revision=rev)


def count_stories(path, sep=b"<|endoftext|>"):
    n, tail = 0, b""
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            data = tail + block
            n += data.count(sep)
            tail = data[-(len(sep) - 1):]  # a separator may straddle two reads
    return n


if __name__ == "__main__":
    for split in ("valid", "train"):
        path = tinystories(split)
        print(f"{os.path.basename(path):40s} {os.path.getsize(path):>15,d} bytes, "
              f"{count_stories(path):,} <|endoftext|> separators")
    path = opus_en_vi()
    print(f"{os.path.basename(path):40s} {os.path.getsize(path):>15,d} bytes")
