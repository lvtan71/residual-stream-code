# Sequences Before Attention: the full code

The scripts behind [Sequences Before Attention: RNNs, Vanishing Gradients, and Why Transformers Won](https://residual-stream.lvtan71.workers.dev/posts/rnns-and-why-transformers-won/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `bptt.py` | An Elman RNN with a 2-number state run for 3 steps, then backpropagation through time by hand, checked against finite differences. | CPU, instant |
| `vanishing.py` | How much of a gradient survives 1, 10, 25 and 50 steps back through a 64-wide recurrence, by the recurrent matrix's scale, linear and with tanh. | CPU, instant |
| `gates.py` | The gradient reaching the input 1 to 50 steps back in untrained RNN, GRU and LSTM cells, and how the GRU's update bias and the LSTM's forget bias change it. | CPU, seconds |
| `corpus.py` | The training and validation text: the blog's English posts as of 2026-09-30, two posts held out. Reports their size and character set. | CPU, instant |
| `ngram.py` | Count models of the previous 0 to 4 characters with add-one smoothing, scored on the held-out posts. | CPU, seconds |
| `kneser_ney.py` | Interpolated Kneser–Ney count models of the previous 1 to 7 characters, and a check that their probabilities sum to one. | CPU, about 5 seconds |
| `charlm.py` | One character-level recurrent model of the blog (helper, run by the next two): `python charlm.py <RNN\|GRU\|LSTM> <seed> [steps]`. | CPU |
| `char_models.py` | Runs `charlm.py` for each cell with seeds 0, 1 and 2, at 3,000 steps. | CPU, about 24 minutes |
| `char_models_long.py` | Runs `charlm.py` for each cell with seed 0 at 12,000 steps. | CPU, about 30 minutes |
| `seq2seq.py` | An encoder-decoder GRU that reverses random digits, with or without attention (helper, run by the next four): `python seq2seq.py <length> <attention\|none> <seed> [steps] [learning rate]`. | CPU |
| `reverse.py` | Runs `seq2seq.py` at lengths 5, 10, 20 and 40, with and without attention, seeds 0 and 1, at 2,000 steps. | CPU, about 20 minutes (estimate) |
| `reverse_40_longer.py` | Length 40, both models, 3,000 steps at the original learning rate. | CPU, about 10 minutes (estimate) |
| `reverse_40_lr.py` | Length 40, both models, 3,000 steps at learning rate 1e-3. | CPU, about 10 minutes (estimate) |
| `reverse_40_6000.py` | Length 40, without attention, 6,000 steps. | CPU, about 6 minutes (estimate) |
| `parallel.py` | One 1,024-token sequence through a GRU (1,024 sequential steps) and through causal self-attention (one pass): FLOPs and time. | CPU, seconds; time it with nothing else running |

`blog_train.txt` and `blog_val.txt` link to `../../data/`: the frozen snapshot `corpus.py` wrote on 2026-09-30, which later posts train on too. With the links in place `corpus.py` reads them instead of rebuilding them. The whole folder runs in about 100 minutes on a CPU. The times for `char_models*.py` add up what `charlm.py` printed; the seq2seq runs are estimates from the total. The scripts need Python 3.12 or later (`kneser_ney.py` uses a backslash inside an f-string). The outputs here came from an Apple M4 laptop with torch 2.11.0 and Python 3.12.13.

```bash
pip install "numpy==2.*" "torch==2.11.*"
python bptt.py
python vanishing.py
python gates.py
python corpus.py
python ngram.py
python kneser_ney.py
python char_models.py
python char_models_long.py
python reverse.py
python reverse_40_longer.py
python reverse_40_lr.py
python reverse_40_6000.py
python parallel.py
```
