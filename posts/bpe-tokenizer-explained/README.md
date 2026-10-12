# Byte-Pair Encoding: the full code

The scripts behind [Byte-Pair Encoding: How Text Becomes Tokens, One Merge at a Time](https://residual-stream.lvtan71.workers.dev/posts/bpe-tokenizer-explained/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `bpe.py` | Byte-level BPE from scratch: pre-tokenizer, naive training, encoding, decoding, special tokens. Imported by the rest. | `regex` |
| `toy.py` | Trains on the 19-word corpus from Sennrich et al. and prints every table, merge and encoding step. | instant |
| `utf8.py` | Characters as UTF-8 bytes, and every code point checked for bytes below 0x80 inside a multi-byte character. | instant |
| `tiktoken_compare.py` | OpenAI's GPT-2, cl100k_base and o200k_base tokenizers on the post's examples. | `tiktoken`, downloads its vocabularies |
| `files.py` | GPT-2's byte-to-character table, and saving and loading `vocab.json` and `merges.txt`. | instant |
| `data.py` | Downloads TinyStories (2.2 GB) and the OPUS-100 en-vi test split at pinned revisions. | `huggingface_hub` |
| `fast_bpe.py` | The same training made fast: parallel reading cut at `<\|endoftext\|>`, an index of which pre-tokens hold each pair, and an optional heap. | |
| `check.py` | Checks the fast trainer against the naive one and the heap against the scan, and that 1, 2, 4 and 8 workers count the same pre-tokens. | about 30 s |
| `no_regex.py` | The first 20 merges on 1 MB of TinyStories with and without the pre-tokenizer. | a few seconds |
| `tinystories.py` | Trains a 10,000-token vocabulary on `valid` (22 MB) or `train` (2.2 GB), timing each stage with a scan and with a heap, and writes `ts_<split>_vocab.json` and `ts_<split>_merges.txt`. | `train`: about 2 minutes and 1 GB per worker; time it with nothing else running |
| `analyze.py` | What the two TinyStories tokenizers learned, and their compression on 200 held-out stories against GPT-2's. | reads `ts_*` |
| `languages.py` | 2,000 English-Vietnamese sentence pairs counted with four tokenizers. | `pandas`, `pyarrow`, reads `ts_train_*` |

The `ts_*` tokenizer files are committed, so `analyze.py` and `languages.py` run without the 2.2 GB training. The outputs came from an Apple M4 laptop (10 cores) with Python 3.12, regex 2026.9.29 and tiktoken 0.14.0; the TinyStories timings were taken with nothing else running.

```bash
pip install regex tiktoken huggingface_hub pandas pyarrow
python toy.py
python utf8.py
python tiktoken_compare.py
python files.py
python data.py
python check.py
python no_regex.py
python tinystories.py valid
python tinystories.py train
python analyze.py
python languages.py
```
