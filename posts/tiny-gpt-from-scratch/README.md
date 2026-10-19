# Build a Tiny GPT, End to End, and Train It on This Blog: the full code

The scripts behind [Build a Tiny GPT, End to End, and Train It on This Blog](https://residual-stream.lvtan71.workers.dev/posts/tiny-gpt-from-scratch/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `params.py` | Counts the model's parameters by part (embedding, attention, MLP, norms) and checks the formula 12 L D² + V D, here and for GPT-2 small. | CPU, instant |
| `sweep.py` | The first round of choices on the dev slice: characters or BPE tokens, and dropout 0.1 to 0.4, one seed each. Runs `train.py` once per configuration. | GPU, about 70 min (28,000 steps at about 0.15 s each on the M4) |
| `ablate.py` | The second round, two seeds each on the dev slice: the chosen model, a cosine schedule, and the model without position information, with learned positions, post-norm, and without residual connections. | GPU, about 2 hours (48,000 steps) |
| `collapse.py` | The models with and without residual connections for 400 steps: training loss, the gradient at each layer's attention weights, and whether the output still depends on the input. | CPU, a few minutes |
| `collapse_lr.py` | The model without residual connections for 1,000 steps at three learning rates, one without dropout: does it still collapse? | CPU, roughly 15–30 min |
| `low_lr.py` | The model without residual connections and the baseline, both at a learning rate of 1e-4, two seeds each, on the dev slice. | GPU, about 40 min (16,000 steps) |
| `score_once.py` | The scored runs: `final.py` with three seeds and the cosine schedule, trained on all the training posts and scored on the two held-out posts. Seed 0 saves `gpt_250.pt` to `gpt_4000.pt`. | GPU, about 30 min (12,000 steps) |
| `per_post.py` | Each held-out post scored separately by the GPT's checkpoints, the LSTM, Kneser–Ney and bz2, plus the GPT on training text it has seen. Reads the `gpt_*.pt` checkpoints and `char_lstm.pt`. | CPU, several minutes |
| `repeats.py` | How much of the held-out text is copied from training or repeated from beyond the model's window, and the model's cost on each kind. Reads `gpt_4000.pt`. | CPU, a few minutes |
| `less_data.py` | `datafrac.py` on a quarter and on half of the training text, two seeds each. | GPU, about 40 min (16,000 steps) |
| `mfu.py` | Times training steps and reports model FLOPs utilization against the M4's measured peak of 2,878 GFLOP/s. On another GPU the times differ and the MFU line needs that GPU's own peak. | GPU, under a minute. Time it with nothing else using the GPU |
| `step_parts.py` | Times a step's parts separately: the matrix multiplies, attention's fused call with and without dropout, and AdamW's update. | GPU, under a minute. Time it with nothing else using the GPU |
| `generate.py` | Samples from seed 0's checkpoints at 250, 1,000 and 4,000 steps, and finds the longest stretch of each sample that appears in the training posts. Reads `gpt_250.pt`, `gpt_1000.pt`, `gpt_4000.pt`. | CPU, a minute or two |

Helpers, imported by the scripts above: `gpt.py` is the model; `train.py` holds the data splits, the tokenizers, the training loop and the scoring, and runs one configuration as `python train.py <name> '<json of run() arguments>'`; `final.py` and `datafrac.py` are one scored run each, with their arguments on the command line. `kn.py` is the Kneser–Ney model from [Sequences Before Attention](https://residual-stream.lvtan71.workers.dev/posts/rnns-and-why-transformers-won/) (its first 40 lines, the definitions), and `lm.py` is the LSTM from [What "Generative" Means](https://residual-stream.lvtan71.workers.dev/posts/autoregression-and-perplexity/); `python lm.py char 0` trains that post's character-level LSTM again (seed 0) and saves it as `char_lstm.pt`, which `per_post.py` reads. The LSTM row of `per_post.out` came from that post's original checkpoint (its label was renamed from "LSTM (post 9, seed 0)" after the run; the numbers are as printed); the retrained one is unverified and can differ slightly.

`blog_train.txt` and `blog_val.txt` link to `../../data/`: the blog corpus that Sequences Before Attention froze, the English posts as of 2026-09-30 with two posts (the KV cache and BPE posts) held out, 593,616 training and 56,786 validation characters.

The times above are estimates from the step counts and the measured 0.15 s per step on the M4; the whole folder is about five hours of GPU time. The outputs here came from an Apple M4 laptop with torch 2.14.0 and tokenizers 0.23.2. `collapse_lr.out` was retyped from the post; the original output file was not kept. Seeds are fixed. On the M4 (mps) the baseline and cosine-schedule runs reproduce the post exactly (other hardware may differ slightly), and the ablations (no position, learned positions, no residual) can differ in the third decimal.

```bash
pip install "torch==2.14.*" "tokenizers==0.23.*"
python params.py
python sweep.py
python ablate.py
python collapse.py
python collapse_lr.py
python low_lr.py
python score_once.py
python lm.py char 0   # trains the LSTM from What "Generative" Means again and saves char_lstm.pt (a few minutes)
python per_post.py
python repeats.py
python less_data.py
python mfu.py
python step_parts.py
python generate.py
```
