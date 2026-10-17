# Embeddings and Contrastive Learning: the full code

The scripts behind [Embeddings and Contrastive Learning: Making Similar Mean Close](https://residual-stream.lvtan71.workers.dev/posts/embeddings-and-contrastive-learning/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `split.py` | Helper, no output: splits the blog text into sections and holds 30% of the long ones out for testing; the rest is training text. | |
| `word2vec.py` | Skip-gram with negative sampling on the training text; saves `word2vec.pt` and prints the loss, nearest neighbors and a few cosines. Takes a seed as an optional argument (default 0). | CPU, about 15 seconds |
| `more_seeds.py` | Runs `word2vec.py` with seeds 1 and 2 (saves `word2vec1.pt`, `word2vec2.pt`). | CPU, about 30 seconds |
| `stable.py` | Which nearest neighbors appear in all three seeds, or in two. | CPU, instant |
| `analogies.py` | The "b - a + c lands near d" test on six analogies from the blog's vocabulary. | CPU, instant |
| `cosine_vs_dot.py` | Vector length against word frequency, and the nearest neighbors of "cache" by dot product and by cosine. | CPU, instant |
| `infonce.py` | InfoNCE on three 2-D pairs at four temperatures, and on random embeddings against log B. | CPU, instant |
| `sections.py` | Helper, no output: loads the word2vec vectors; `to_ids` and `crop`, shared by `contrastive.py` and `pretrained.py`. | |
| `contrastive.py` | Baselines, and a contrastive linear head on averaged word2vec vectors at batch 4, 16 and 64, three seeds each: same-section recall@1 on the held-out sections. | CPU, about a minute |
| `pretrained.py` | all-MiniLM-L6-v2 and multilingual MiniLM-L12 on the same test, and on matching English sections to their Vietnamese translations. | CPU, under a minute; downloads both models (about 90 MB and 470 MB) at pinned revisions. Reads `content/`, unpacked from `blog-47b01b1.tar.gz` |

Run times are estimates; the whole folder takes about 2 minutes. `blog_train.txt` and `blog_val.txt` link to `../../data/`: the blog's English posts as plain text, the corpus several posts in the series train on. `blog-47b01b1.tar.gz` links to `../../data/blog-47b01b1.tar.gz`, a snapshot of the blog's posts at commit 47b01b1; `pretrained.py` reads its English posts and their Vietnamese twins.

The outputs here came from an Apple M4 laptop. `pretrained.py` ran with torch 2.14.0, transformers 5.17.0 and sentence-transformers 6.1.0; the other scripts' outputs came from torch 2.11.0 (Python 3.12.13), except `more_seeds.out`, which came from a rerun with torch 2.14.1. Rerun with torch 2.14.1, every script up to `contrastive.py` printed the same output byte for byte.

```bash
pip install "torch==2.14.*" "transformers==5.17.*" sentence-transformers
tar xzf blog-47b01b1.tar.gz
python word2vec.py
python more_seeds.py
python stable.py
python analogies.py
python cosine_vs_dot.py
python infonce.py
python contrastive.py
python pretrained.py
```
