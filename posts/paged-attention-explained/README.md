# Paged Attention: the full code

The complete scripts behind [Paged Attention: Managing the KV Cache Like Virtual Memory](https://residual-stream.lvtan71.workers.dev/posts/paged-attention-explained/). The post shows the important parts of each script, and each `.out` file is what the script printed for the post.

Run them from this folder, in this order:

| Script | What it does | Needs |
| --- | --- | --- |
| `waste.py` | How much of a KV cache holds real tokens under four ways of reserving memory, on ShareGPT and Alpaca request lengths. Writes `sharegpt_lengths.json` and `alpaca_lengths.json` (both included here). | CPU; downloads ShareGPT (673 MB) and Alpaca (24 MB) from Hugging Face at pinned revisions |
| `paged.py` | The paged KV cache: a block pool, block tables, and copy-on-write, plugged into Hugging Face's cache interface. Imported by the next two. | |
| `same.py` | Two prompts decoded through the paged cache, checked token for token against the contiguous cache. | GPU or CPU; downloads Qwen2.5-1.5B-Instruct (about 3 GB) |
| `share.py` | Four samples of one prompt sharing its blocks. | GPU or CPU |
| `prefix.py` | Prefix caching by chained block hashes. | CPU |
| `blocksize.py` | Waste and table length against block size. Reads the two length files. | CPU |
| `kernel.py` | Decode attention through a block table, as a Metal kernel, timed against a contiguous cache. | Apple-silicon Mac only (Metal, through `torch.mps.compile_shader`); time it with nothing else using the GPU |

`lm.py` loads the model and tokenizer; the other scripts import it. The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.19.0 (`blocksize.py`, plain Python, with the system Python 3.9.6).

```bash
pip install "torch==2.14.*" "transformers==5.19.*" huggingface_hub pandas pyarrow
python waste.py
python same.py
python share.py
python prefix.py
python blocksize.py
python kernel.py       # Apple-silicon Mac only (Metal)
```
