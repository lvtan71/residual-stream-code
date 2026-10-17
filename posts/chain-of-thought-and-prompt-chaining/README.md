# Chain-of-Thought and Prompt Chaining: the full code

The scripts behind [Chain-of-Thought and Prompt Chaining](https://residual-stream.lvtan71.workers.dev/posts/chain-of-thought-and-prompt-chaining/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `serial_steps.py` | 40 serial arithmetic problems for each length k in 1, 2, 3, 4, 6, 8, solved six ways: one direct call, one reasoned call, a chain answering directly or reasoning first, and the direct chain checked by agreement or by an oracle (three seeds each). Writes every greedy chained step to `chain_steps.json` and reports which steps fail. | GPU, about 65 minutes; downloads Qwen2.5-1.5B-Instruct (about 3 GB) at a pinned revision |
| `step_errors.py` | Reads `chain_steps.json` and prints the error analysis: wrong steps, how many were multiplications or started from 1,000 or more, accuracy below 1,000, wrong steps around zero, and per-step accuracy on the eight-step problems. | CPU, instant |

`lm.py` loads Qwen2.5-1.5B-Instruct, the model from [Decoding](https://residual-stream.lvtan71.workers.dev/posts/decoding-from-logits-to-text/), and its tokenizer, the same helper as in other posts that use this model. `step_errors.out` holds the lines the original one-off analysis printed (the same code, now `step_errors.py`); `chain_steps.json` from that run was not kept. The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.18.0.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.18.*" huggingface_hub
python serial_steps.py
python step_errors.py
```
