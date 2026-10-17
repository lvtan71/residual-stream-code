# Fine-Tuning, and What It Destroys: the full code

The scripts behind [Fine-Tuning, and What It Destroys](https://residual-stream.lvtan71.workers.dev/posts/fine-tuning-and-forgetting/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `sft_loss.py` | One chat example through Qwen2.5's chat template, with the prompt's tokens masked to -100 so only the answer is trained. | CPU, instant; downloads the Qwen2.5-1.5B-Instruct repository at a pinned revision (about 3 GB; only its tokenizer is used) |
| `fetch_wiki.py` | Downloads one shard each of English and Vietnamese Wikipedia (the November 2023 dump) into `wiki/`. | Network, about 0.7 GB, pinned revision |
| `wiki_data.py` | Builds the replay text (`wiki_replay.txt`, about 4 million characters of English Wikipedia, minus any article whose title is in WikiText-2's test set) and the Vietnamese test text (`vi_test.txt`, 120,000 characters). Reads `wiki/`. | CPU, under a minute; downloads WikiText-2 (a few MB) at a pinned revision |
| `finetune_all.py` | Fine-tunes Qwen2.5-0.5B eight times with `ft.py` (learning rates 1e-5 and 1e-4, with and without 25% replay, two seeds, 400 steps each) and keeps each run's summary line in `ft.out`. It printed those eight lines, which were not kept, so it has no `.out`. | GPU, about 4 hours on the M4 laptop; downloads Qwen2.5-0.5B (about 1 GB) at a pinned revision |
| `ft_table.py` | Turns `ft.out` into the post's table: bits per byte on the held-out blog posts, WikiText-2 and Vietnamese Wikipedia at each checkpoint. | CPU, instant |

`ft.py` is one fine-tuning run, `python ft.py <lr> <replay> <steps> <seed>`: the model with its embedding table frozen, gradient checkpointing, two micro-batches of 2 windows of 512 tokens per step, and bits per byte at steps 0, 25, 50, 100, 200 and 400. [Parameter-Efficient Fine-Tuning](https://residual-stream.lvtan71.workers.dev/posts/parameter-efficient-fine-tuning-lora/) passes a sixth argument, a file to save the trained layers to.

`blog_train.txt` and `blog_val.txt` link to `../../data/blog-2026-10-04/`: the blog's English posts rebuilt as of 2026-10-04 (commit 5a4ae53 plus a draft of one later post), with two posts (the KV cache and BPE posts) held out, 1,747,277 training and 60,790 validation characters.

The outputs here came from an Apple M4 laptop (16 GB) with torch 2.14.1 and transformers 5.18.0 (`sft_loss.out` with torch 2.14.0 and transformers 5.17.0). Seeds are fixed; GPU training is not bit-for-bit deterministic, so the last digit of the table can differ.

```bash
pip install "torch==2.14.*" "transformers==5.18.*" huggingface_hub pandas pyarrow
python sft_loss.py
python fetch_wiki.py
python wiki_data.py
python finetune_all.py
python ft_table.py
```
