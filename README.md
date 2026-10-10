# Residual Stream: runnable code

Notebooks that rerun the code in the posts of
[Residual Stream](https://residual-stream.lvtan71.workers.dev), a notebook on
LLM internals and AI engineering by Tan Le. Each notebook runs a post's code
in reading order and shows, after every cell, the output the post printed.

| Post | Runs on | Notebook |
| --- | --- | --- |
| [The KV Cache: How Transformers Avoid Recomputing the Past](https://residual-stream.lvtan71.workers.dev/posts/kv-cache-explained/) | CPU | [Open in Colab](https://colab.research.google.com/github/lvtan71/residual-stream-code/blob/main/notebooks/kv-cache-explained.ipynb) |
| [Cross-Entropy and the Softmax Gradient: One Loss for Every Classifier](https://residual-stream.lvtan71.workers.dev/posts/softmax-and-cross-entropy/) | CPU | [Open in Colab](https://colab.research.google.com/github/lvtan71/residual-stream-code/blob/main/notebooks/softmax-and-cross-entropy.ipynb) |
| [Speculative Decoding, Derived: Draft Models, N-grams, Medusa and EAGLE](https://residual-stream.lvtan71.workers.dev/posts/speculative-decoding/) | GPU | [Open in Colab](https://colab.research.google.com/github/lvtan71/residual-stream-code/blob/main/notebooks/speculative-decoding.ipynb) |

More posts are being added.

## Running locally

```bash
git clone https://github.com/lvtan71/residual-stream-code
cd residual-stream-code
pip install numpy torch transformers jupyter
jupyter lab notebooks/
```

Each notebook's first cell copies the helpers and data it needs into
`work/<post>/` and runs there. In Colab, the same cell clones this repo first.

## What's here

- `notebooks/`: one notebook per post.
- `shared/`: helper files that posts ask readers to save (`router_data.py`,
  `softmax_basics.py`, ...), so a post can import what an earlier post built.
- `data/`: the data the posts use.
  - `router-questions.csv`: the question router's 220 labeled questions.
  - `blog_train.txt`, `blog_val.txt`: the blog's English posts as of
    2026-09-30 as plain text. Two posts are held out for validation. Several
    posts train language models on this text.

The notebooks are generated from the posts, so the code here is the code in
the post. Please report a mismatch as an issue.

The posts' numbers come from an Apple M4 laptop. On other hardware, timings
and the last digits of some floating-point results will differ.
