# Watermarking LLM Text: the full code

The scripts behind [Watermarking LLM Text: A Signal Hidden in the Sampling](https://residual-stream.lvtan71.workers.dev/posts/watermarking-llm-text/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `z_test.py` | The detector's z-test worked through for γ = 0.25, δ = 2: the green share of a flat distribution, z at 50 and 200 tokens, tokens needed for z = 4, and the false-positive rate at z > 4 from the normal tail and the exact binomial. | Python, instant |
| `wm.py` | The Kirchenbauer et al. watermark on Qwen2.5-1.5B-Instruct: 10 prompts, 200 tokens each, at δ = 0, 1, 2 and 4; green share, z, detections and quality (NLL). Then z against entropy at δ = 2, and two attacks: tokens replaced at random, and a paraphrase by the same model. | GPU, roughly 10 minutes (an estimate: about 10,000 generated tokens); downloads Qwen2.5-1.5B-Instruct (about 3 GB) at a pinned revision |
| `entropy_paraphrase.py` | The δ = 2 texts generated again with a log of every step: the green rate by the step's entropy, and how many token pairs each paraphrase shares with its original. Takes the model and functions from `wm.py`. | GPU, roughly 4 minutes (an estimate) |
| `human.py` | False positives on human text: every 200-token window of the blog's posts scored with `wm.py`'s key, counting every pair and each distinct pair once. Loads only the tokenizer. | CPU, a minute or two (an estimate) |
| `other_keys.py` | The same windows, distinct pairs only, scored with five other keys: is the mean below zero a property of one key? Takes its code from `human.py`. | CPU, several minutes (an estimate) |

`entropy_paraphrase.py` takes its definitions from the top of `wm.py`, and `human.py` and `other_keys.py` from `wm.py` and `human.py`, by reading those files and running the part before their experiments, so that importing them doesn't run the experiments too.

`blog-fb57e84.tar.gz` links to `../../data/`: a snapshot of the blog's source at commit fb57e84, the 52 English posts as of the post's run; `tar xzf` unpacks its `content/posts/`, which `human.py` and `other_keys.py` read.

The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17.0. Seeds are fixed, but sampling on another GPU can choose different tokens, so the numbers from `wm.py` and `entropy_paraphrase.py` can differ in detail.

```bash
pip install numpy "torch==2.14.*" "transformers==5.17.*" huggingface_hub
tar xzf blog-fb57e84.tar.gz
python z_test.py
python wm.py
python entropy_paraphrase.py
python human.py
python other_keys.py
```
