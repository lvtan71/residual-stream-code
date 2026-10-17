# Cross-Entropy and the Softmax Gradient: the full code

The scripts behind [Cross-Entropy and the Softmax Gradient: One Loss for Every Classifier](https://residual-stream.lvtan71.workers.dev/posts/softmax-and-cross-entropy/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `one_vs_rest.py` | The four-way softmax router against four independent sigmoids (one-vs-rest) on 20 random splits: validation accuracy, how far the sigmoid scores are from summing to 1, and validation cross-entropy. | CPU, a few seconds |
| `info.py` | Entropy, cross-entropy and KL divergence for four models of one question's route distribution; ln 4 and ln 128,256; the logit gap and loss floor that label smoothing sets. | CPU, instant |
| `softmax_basics.py` | `softmax` and `cross_entropy` from logits (the helper the other scripts import); run on its own, it checks the gradient p - y against finite differences and shows why the max is subtracted first. | CPU, instant |
| `four_routes.py` | L2, no penalty, label smoothing and both, on 20 splits, paired against L2; then the model chosen on validation cross-entropy, scored on the pooled test splits with a confusion matrix, precision, recall and F1. | CPU, a few seconds |
| `longer_training.py` | The same four models trained for 4,000 and 16,000 steps: validation and training cross-entropy, weight norm, mean top probability. | CPU, about 45 seconds |
| `read_weights.py` | One split's learned weights per route, its bias, and the two architecture questions it sends to foundations, word by word. | CPU, instant |
| `unsure.py` | Across 20 splits: how often the router's top probability is below 0.4, how often those questions are right, where they go, and whether an even bias would move them. | CPU, a few seconds |
| `logit_memory.py` | Memory for the logits of one training batch at Llama 3 8B's 128,256-token vocabulary, against the hidden states and checkpointed activations. | Python, instant |

Helpers: `softmax_basics.py` (above); `router4.py`, the four-way bag of words, the training loop with label smoothing, and the top-label calibration error; and `router_data.py`, the question set, splits and bag of words from [Features and Overfitting](https://residual-stream.lvtan71.workers.dev/posts/features-overfitting-regularization/) (a copy).

`router-questions.csv` links to `../../data/router-questions.csv`: the hand-labeled reader questions, 50 English questions per route (plus 20 Vietnamese ones the scripts here do not use).

The whole folder runs in about a minute on a CPU (the per-script times are estimates). The outputs here came from an Apple M4 laptop; the scripts need only numpy.

```bash
pip install "numpy==2.*"
python one_vs_rest.py
python info.py
python softmax_basics.py
python four_routes.py
python longer_training.py
python read_weights.py
python unsure.py
python logit_memory.py
```
