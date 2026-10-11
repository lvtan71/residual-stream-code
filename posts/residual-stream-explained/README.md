# The Residual Stream: the full code

The scripts behind [The Residual Stream: How a Transformer Turns Tokens Into Predictions](https://residual-stream.lvtan71.workers.dev/posts/residual-stream-explained/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `softmax.py` | Softmax on the post's toy logits; shifting and scaling them. | numpy, instant |
| `params.py` | Where a block's weights and FLOPs go (Llama 1 7B and Llama 3 8B widths), and Qwen2.5-1.5B-Instruct's weights by role. | torch and transformers; downloads only a config |
| `stream.py` | Qwen2.5-1.5B-Instruct's residual stream on 512 tokens of this blog: its size, each block's updates, the logit lens, and each block removed in turn. | GPU, under a minute (slower on a CPU, where 16-bit results can differ slightly); downloads the model (3 GB) at a pinned revision. Reads `blog_train.txt`, a link to `../../data/blog_train.txt` |

`lm.py` loads Qwen2.5-1.5B-Instruct and its tokenizer. The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.19. `stream.py`'s hooks rely on transformers 5's Qwen2 layers returning a plain tensor (version 4 returned a tuple), hence the pins.

```bash
pip install numpy "torch==2.14.*" "transformers==5.19.*" huggingface_hub
python softmax.py
python params.py
python stream.py
```
