# Long Context, and Where It Breaks: the full code

The scripts behind [Long Context, and Where It Breaks: Lost in the Middle](https://residual-stream.lvtan71.workers.dev/posts/long-context-lost-in-the-middle/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `position.py` | One task (`kv` or `needle`) at four prompt lengths and five depths: a JSON object of random UUID pairs and one key to look up, or a passphrase hidden in the blog's prose. Prints the trials right per depth. Optional arguments `<sizes> <trials> <salt>` give more trials with new draws. | GPU; downloads Qwen2.5-1.5B-Instruct (about 3 GB) at a pinned revision. Reads `blog_train.txt` |
| `both_tasks.py` | Runs `position.py kv`, then `position.py needle`. | GPU, about 35 minutes (24 and 10) |
| `more_trials.py` | Runs `position.py kv 120,240 12 -b`: 12 new trials at 8,000 tokens and 12 at 16,000. | GPU, about 41 minutes |
| `key_first.py` | The 12 objects of the 8,100-token, 50%-depth cell, regenerated from the same seed, asked with the key after the data only and with it before and after; every answer classified. | GPU, a few minutes (estimate) |
| `dilution.py` | Softmax dilution: the attention share of one key that scores a fixed margin above n - 1 equal others. | Python, instant |
| `sinks.py` | Attention weights of a 1,024-token passage: the share on the first token, the last four and the rest, per layer and per head, and which position gets the most attention when the first token changes. | GPU, a few minutes (estimate); 32-bit eager attention |

`lm.py` loads Qwen2.5-1.5B-Instruct and its tokenizer (the model on first use) and picks the device; it is adapted from the helper of [Decoding](https://residual-stream.lvtan71.workers.dev/posts/decoding-from-logits-to-text/), which loads the same model at the same revision. `blog_train.txt` links to `../../data/blog_train.txt`, the frozen snapshot of the blog's posts written by [Sequences Before Attention](https://residual-stream.lvtan71.workers.dev/posts/rnns-and-why-transformers-won/).

The whole folder takes about 80 minutes on an Apple M4 laptop GPU. The per-script times above are from the post (the "s so far" column); those marked as estimates are the remainder of the 80 minutes. The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17.

```bash
pip install "torch==2.14.*" "transformers==5.17.*" huggingface_hub
python both_tasks.py
python more_trials.py
python key_first.py
python dilution.py
python sinks.py
```
