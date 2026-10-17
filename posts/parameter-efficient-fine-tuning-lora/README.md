# Parameter-Efficient Fine-Tuning: the full code

The scripts behind [Parameter-Efficient Fine-Tuning: LoRA, Prefix and Prompt Tuning](https://residual-stream.lvtan71.workers.dev/posts/parameter-efficient-fine-tuning-lora/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `memory.py` | LoRA's trainable-parameter count and training memory for Llama 3.1 8B: full fine-tune, LoRA and QLoRA at rank 16, and the size of one adapter. Plain arithmetic. | CPU, instant |
| `fetch_wiki.py` | Downloads one shard each of English and Vietnamese Wikipedia (the November 2023 dump) into `wiki/`. | Network, about 0.7 GB, pinned revision |
| `wiki_data.py` | Builds `wiki_replay.txt` and `vi_test.txt` from `wiki/`, which `ft.py` reads. | CPU, under a minute; downloads WikiText-2 (a few MB) at a pinned revision |
| `full_ft.py` | Reruns the full fine-tune of "Fine-Tuning, and What It Destroys" at lr 1e-5 without replay, seeds 0 and 1, with `ft.py`; seed 0 saves its block weights to `ft_weights.pt`. Keeps each run's summary line in `full_ft.log`. Its printed lines were not kept, so it has no `.out`. | GPU, about an hour (two runs of 25 to 35 minutes); downloads Qwen2.5-0.5B (about 1 GB) at a pinned revision |
| `lora_runs.py` | Runs `lora.py` eleven times: ranks 1, 8 and 64 at 1e-3 and rank 8 at 2e-4, seeds 0 and 1 (`lora.log`), then rank 64 at 1.25e-4 twice and rank 1 at 2e-4 once (`lora_extra.log`). No `.out`, as above. | GPU, about 5 to 6 hours (1,481 to 2,100 seconds a run on the M4) |
| `lora_table.py` | Turns `lora.log`, `lora_extra.log` and `full_ft.log` into the post's table of bits per byte at each checkpoint, the run times, and the merged-vs-unmerged logit check. | CPU, instant |
| `spectrum.py` | For each block matrix of the full fine-tune, the share of the update's squared size captured by its best rank-1, 8 and 64 approximations, the rank needed for 90%, and the same for a random matrix. Arguments: the weights file and a label. | CPU, a few minutes; Qwen2.5-0.5B |
| `truncated_update.py` | Rebuilds the fine-tuned model with each matrix's update cut to rank 0, 1, 8 and 64, and measures held-out blog and WikiText bits per byte. | GPU, a few minutes; Qwen2.5-0.5B |

`lora.py` is one LoRA run, `python lora.py <rank> <lr> <seed>`: Qwen2.5-0.5B with every pretrained weight frozen and a rank-r adapter (alpha = 2r) on the seven linear layers of each block, trained 400 steps on the blog text, measured at steps 0, 25, 50, 100, 200 and 400, then merged and checked. `ft.py`, `fetch_wiki.py` and `wiki_data.py` are copies from the folder of [Fine-Tuning, and What It Destroys](https://residual-stream.lvtan71.workers.dev/posts/fine-tuning-and-forgetting/), and `wiki_data.out` is what `wiki_data.py` printed there.

`blog_train.txt` and `blog_val.txt` link to `../../data/blog-2026-10-04/`: the blog's English posts rebuilt as of 2026-10-04, with two posts (the KV cache and BPE posts) held out.

Run times other than the LoRA runs' are estimates. The outputs here came from an Apple M4 laptop (16 GB) with torch 2.14.1 and transformers 5.18.0; `memory.py` ran with the system python3 (Python 3.9.6), as it needs no libraries. Seeds are fixed; GPU training is not bit-for-bit deterministic, so the last digit of the table can differ.

```bash
pip install "torch==2.14.*" "transformers==5.18.*" huggingface_hub pandas pyarrow
python memory.py
python fetch_wiki.py
python wiki_data.py
python full_ft.py
python lora_runs.py
python lora_table.py
python spectrum.py ft_weights.pt "lr 1e-5, no replay, 400 steps, seed 0"
python truncated_update.py
```
