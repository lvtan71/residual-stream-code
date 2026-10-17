# Running LLMs on a Laptop: the full code

The scripts behind [Running LLMs on a Laptop: GGUF and llama.cpp](https://residual-stream.lvtan71.workers.dev/posts/running-llms-on-a-laptop/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `gguf_contents.py` | What is inside a GGUF file: Qwen2.5-1.5B-Instruct Q4_K_M's metadata, its tensors grouped by quantization type, which tensors are kept at Q6_K, and the average bits per weight of the Q3_K_M, Q4_K_M and Q8_0 files. | CPU, a few seconds; downloads the three GGUF files (about 4 GB together) at a pinned revision |
| `speed.py` | Decode speed of the three files on the GPU (all layers offloaded) and on 8 CPU threads, median of 3 runs of 128 tokens, against the bandwidth bound of 93 GB/s. | GPU, under a minute. The post's GPU numbers are from Apple's Metal; with a CPU-only build of llama-cpp-python, `n_gpu_layers=-1` runs on the CPU, so expect the bound and the CPU column to differ and the GPU column to be meaningless. Time it with nothing else using the machine |
| `quality.py` | Bits per token of each file on two of the blog's posts, from llama.cpp's own logits. | GPU, under a minute. Reads `content/posts/`, unpacked from `blog-ed51ca8.tar.gz` |

`corpus.py` is the corpus code of [Sequences Before Attention](https://residual-stream.lvtan71.workers.dev/posts/rnns-and-why-transformers-won/), in the earlier version this post ran with; `quality.py` imports its `clean`. `blog-ed51ca8.tar.gz` links to `../../data/blog-ed51ca8.tar.gz`, a snapshot of the blog's posts at commit ed51ca8.

The whole folder takes about a minute after the downloads. The outputs here came from an Apple M4 laptop with llama-cpp-python 0.3.36 built with Metal.

```bash
pip install llama-cpp-python gguf huggingface_hub numpy
tar xzf blog-ed51ca8.tar.gz
python gguf_contents.py
python speed.py
python quality.py
```
