# What "Generative" Means: the full code

The scripts behind [What "Generative" Means: Autoregression, the Chain Rule, and Perplexity](https://residual-stream.lvtan71.workers.dev/posts/autoregression-and-perplexity/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `bigram.py` | The chain rule on "the cache" with a character bigram model counted on the training posts (add-one smoothing), and the bigram model's perplexity on the held-out posts. | CPU, instant |
| `train_all.py` | Trains the LSTM language model of `lm.py` over characters and over byte-level BPE tokens, two seeds each, and prints held-out bits per byte at five checkpoints. The character model with seed 0 is saved as `char_lstm.pt`. | CPU, about 20 minutes (four runs of 3,000 steps) |
| `generate.py` | Greedy and sampled continuations of "The KV cache " from `char_lstm.pt`, and the time to score 512 known characters in one call, one call per character, and to generate 512. Reads `char_lstm.pt`. | CPU, under a minute |
| `baselines.py` | Bits per byte on the held-out posts for raw bytes, a uniform guess, unigram counts, and zlib, bz2 and xz with and without the training posts. | CPU, a few seconds |

`lm.py` is the helper: the BPE tokenizer trained on the training posts, the LSTM model, the training loop and the scoring. `python lm.py <char|bpe> <seed>` trains and scores one model; `train_all.py` runs it four times.

`blog_train.txt` and `blog_val.txt` link to `../../data/`: the blog corpus that [Sequences Before Attention](https://residual-stream.lvtan71.workers.dev/posts/rnns-and-why-transformers-won/) froze, the English posts as of 2026-09-30 with two posts (the KV cache and BPE posts) held out, 593,616 training and 56,786 validation characters.

The whole folder runs in about 22 minutes on a CPU; the per-script times are estimates from that total. The outputs here came from an Apple M4 laptop with torch 2.14.0 and tokenizers 0.23.2. Seeds are fixed; the timings in `generate.out` depend on the machine.

```bash
pip install "torch==2.14.*" "tokenizers==0.23.*"
python bigram.py
python train_all.py
python generate.py
python baselines.py
```
