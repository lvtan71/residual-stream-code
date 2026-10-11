# The Transformer's Three Attention Patterns: the full code

The scripts behind [The Transformer's Three Attention Patterns: Self, Masked, and Cross](https://residual-stream.lvtan71.workers.dev/posts/transformer-attention-patterns/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `attention.py` | The three patterns as one function, on the post's worked examples; then permutation equivariance. | numpy, instant |
| `scale.py` | Why attention divides by √d: the spread of q·k and what it does to a softmax. | numpy, seconds |
| `causal.py` | Qwen2.5-1.5B-Instruct: the causal mask hides the future exactly; the strongest previous-token and first-token heads. Writes `heads.json`. | GPU or CPU, a minute; downloads the model (3 GB) at a pinned revision |
| `cross.py` | Marian English-to-German: cross-attention from each German token to the English ones, by layer. Writes `cross.json`. | CPU, seconds; downloads the model (300 MB) at a pinned revision |
| `cost.py` | Causal attention at Qwen2.5-1.5B's shapes, timed from 512 to 8,192 tokens. | GPU; time it with nothing else using the GPU |

The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.19.

```bash
pip install numpy torch transformers huggingface_hub sentencepiece
python attention.py
```
