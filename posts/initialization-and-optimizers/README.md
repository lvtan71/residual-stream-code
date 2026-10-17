# Initialization and Optimizers: the full code

The scripts behind [Initialization and Optimizers: Where to Start, and How to Step](https://residual-stream.lvtan71.workers.dev/posts/initialization-and-optimizers/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `depth_signal.py` | Activation and gradient size through 20 layers of width 256 for five initializations, median of five seeds. | CPU, about a second |
| `deep_init.py` | The four-way router with ten hidden ReLU layers, trained from three initializations with SGD and Adam: first-layer gradient and loss at 0, 100 and 1,000 steps. Its `init` is imported by `adam_eps.py`. | CPU, about 10 seconds |
| `momentum.py` | Plain gradient descent against heavy-ball momentum on the tokens-against-words fit of Learning Is Optimization: steps to converge, against theory. | CPU, instant |
| `bias_correction.py` | Adam's moving averages on a constant gradient, with and without bias correction, and the first step for gradients of very different sizes. | CPU, instant |
| `adam_eps.py` | The deep router from the tiny start, trained by Adam with epsilon 1e-8 and 1e-30, five seeds. | CPU, about 5 seconds |
| `race.py` | The four-way linear router with an L2 penalty, trained full-batch by gradient descent, momentum and Adam over a grid of learning rates; the best objective at 30, 300 and 3,000 steps. Add the argument `length` for the raw question-length feature. Its `run` is imported by `race_curves.py`. | CPU, about 15 seconds per call |
| `race_both.py` | Runs `race.py`, then `race.py length`. | CPU, about 35 seconds |
| `curvature.py` | The largest curvature of the race's objective at the start, without and with the length feature, and the step size it allows. | CPU, instant |
| `race_curves.py` | The objective curves behind the post's chart: each optimizer at its best learning rate, median of five splits. | CPU, about 3 seconds |
| `warmup_noise.py` | Adam on pure-noise gradients for 100,000 parameters: the typical step and the share of runaway per-parameter rates, by step. | CPU, about 3 seconds |
| `decay.py` | No decay, SGD with L2, Adam with L2 and AdamW on the router, mini-batches of 16, 20 splits: validation choice, weight sizes, test results. The argument sets the steps per run (default 3,000). | CPU, about 25 seconds; with `12000`, about 2 minutes |
| `adam_memory.py` | Bytes of training state per parameter for an 8B model under mixed precision. | Python, instant |

Helpers: `optim.py` (SGD, momentum, Adam and AdamW) and `deep.py` (a ReLU network's loss and gradients) are this post's. `a_loss.py` is from [Learning Is Optimization](https://residual-stream.lvtan71.workers.dev/posts/learning-is-optimization/), `router_data.py` from [Features and Overfitting](https://residual-stream.lvtan71.workers.dev/posts/features-overfitting-regularization/), and `softmax_basics.py` and `router4.py` from [Cross-Entropy and the Softmax Gradient](https://residual-stream.lvtan71.workers.dev/posts/softmax-and-cross-entropy/). `router-questions.csv` links to `../../data/router-questions.csv`, the hand-labeled reader questions.

The whole folder runs in about 3 minutes on a CPU. The outputs here came from an Apple M4 laptop; the scripts need only numpy.

```bash
pip install "numpy==2.*"
python depth_signal.py
python deep_init.py
python momentum.py
python bias_correction.py
python adam_eps.py
python race_both.py
python curvature.py
python race_curves.py
python warmup_noise.py
python decay.py
python decay.py 12000
python adam_memory.py
```
