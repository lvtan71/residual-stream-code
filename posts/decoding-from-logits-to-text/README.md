# Decoding: the full code

The scripts behind [Decoding: From a Distribution to Text](https://residual-stream.lvtan71.workers.dev/posts/decoding-from-logits-to-text/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `lm.py` | Loads Qwen2.5-1.5B-Instruct and its tokenizer in 16-bit floats. A helper, no output. | Downloads the model (about 3 GB) at a pinned revision |
| `sampling.py` | The decoding policies as edits to the logits (repetition, frequency and presence penalties, temperature, top-k, top-p, min-p), one step's choice, and a generation loop with a KV cache. A helper, no output. | |
| `distributions.py` | Two real next-token distributions: the most likely tokens, the entropy at three temperatures, and how many tokens each filter keeps, before and after temperature. Writes `widget_dist.json`, the data for the post's widget. | GPU, under a minute |
| `human_logprob.py` | The log-probability of each prompt's real continuation, and whether the paragraphs full of numbers are the hard ones. | GPU, under a minute |
| `compare.py` | Every policy on the 20 prompts, 128 new tokens each, one or two seeds: repeated 4-grams and log-probability, against the real continuations. Writes `compare_texts.json`. | GPU, about 33 minutes (the times in `compare.out`) |
| `one_prompt.py` | One prompt's continuation under each policy, from `compare_texts.json`, with the real continuation last. | CPU, instant |
| `reproducible.py` | The same seed twice, and greedy decoding alone against in a padded batch of four. | GPU, about 2 minutes |
| `det_check.py` | The first step's logits for four prompts, alone and in a padded batch, in 16-bit or 32-bit floats (the argument). | GPU |
| `det_check_both.py` | Runs `det_check.py 16` and `det_check.py 32`. | GPU, about a minute |
| `cost.py` | One forward pass for a token against the time to choose it under three policies. | GPU, under a minute. Time it with nothing else using the GPU |

`prompts.json` links to `../../data/prompts.json`: 20 paragraphs of the blog's prose, each cut into a 200-character prompt and its real continuation. Run times other than `compare.py`'s are estimates from the post's 40 minutes in total. On a GPU other than the post's Apple M4, the 16-bit rounding differences in `reproducible.py` and `det_check_both.py` will differ.

The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17.

```bash
pip install "torch==2.14.*" "transformers==5.17.*" huggingface_hub
python distributions.py
python human_logprob.py
python compare.py
python one_prompt.py
python reproducible.py
python det_check_both.py
python cost.py
```
