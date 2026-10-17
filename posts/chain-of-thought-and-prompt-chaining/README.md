# Chain-of-Thought and Prompt Chaining: the full code

The scripts behind [Chain-of-Thought and Prompt Chaining](https://residual-stream.lvtan71.workers.dev/posts/chain-of-thought-and-prompt-chaining/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `serial_steps.py` | 40 serial arithmetic problems for each length k in 1, 2, 3, 4, 6, 8, solved six ways: one direct call, one reasoned call, a chain answering directly or reasoning first, and the direct chain checked by agreement or by an oracle (three seeds each). Writes every greedy chained step to `chain_steps.json` and reports which steps fail. | GPU, about 65 minutes; downloads Qwen2.5-1.5B-Instruct (about 3 GB) at a pinned revision |

`lm.py` loads Qwen2.5-1.5B-Instruct, the model from [Decoding](https://residual-stream.lvtan71.workers.dev/posts/decoding-from-logits-to-text/), and its tokenizer, the same helper as in other posts that use this model. The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.18.0.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.18.*" huggingface_hub
python serial_steps.py
```
