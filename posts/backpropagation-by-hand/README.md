# Backpropagation, by Hand: the full code

The scripts behind [Backpropagation, by Hand: The Chain Rule, Run Backward](https://residual-stream.lvtan71.workers.dev/posts/backpropagation-by-hand/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `mlp.py` | The network: a ReLU hidden layer and a sigmoid output, its forward pass, backpropagation for the mean log-loss with an L2 penalty, and full-batch gradient descent. A helper, no output. | CPU |
| `xor.py` | The post's hand calculation (one XOR example, every gradient, one step), then how often training finds XOR from 100 random starts at hidden widths 2 to 16. | CPU, about 50 seconds |
| `xor_failures.py` | The 82 two-unit starts that fail, classified: right answers, final loss, whether a hidden unit was dead from step 0, and the gradient's size. | CPU, about 10 seconds |
| `grad_check.py` | Backpropagation against central finite differences on a small random problem. | CPU, instant |
| `router_mlp.py` | The router with 16 hidden units and five L2 penalties against the linear router, on 20 splits, paired differences with the corrected standard error, then the test score of the network chosen on validation. | CPU, about 20 seconds |
| `steps_and_seeds.py` | The network at 3,000 and 12,000 steps and from five initialization seeds per split, with and without the penalty. | CPU, about 2.5 minutes |

`router-questions.csv` links to `../../data/router-questions.csv`, the 220 labeled reader questions (200 English, 20 Vietnamese). `router_data.py` (the splits and bag-of-words features) and `overfit.py` (the linear router's training) are copies of the helpers from [Features and Overfitting](https://residual-stream.lvtan71.workers.dev/posts/features-overfitting-regularization/); `router_models.py` (the keyword count) is a copy of the helper from [Measuring a Classifier Honestly](https://residual-stream.lvtan71.workers.dev/posts/measuring-a-classifier/). Importing `router_models.py` trains that post's routers on 20 splits first, which adds a few seconds.

The whole folder runs in about 4 minutes on a CPU. The outputs here came from an Apple M4 laptop; the scripts need only numpy.

```bash
pip install "numpy==2.*"
python xor.py
python xor_failures.py
python grad_check.py
python router_mlp.py
python steps_and_seeds.py
```
