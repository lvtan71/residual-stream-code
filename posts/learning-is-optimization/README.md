# Learning Is Optimization: the full code

The scripts behind [Learning Is Optimization: Linear and Logistic Regression, Trained by Gradient Descent](https://residual-stream.lvtan71.workers.dev/posts/learning-is-optimization/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `a_loss.py` | Words and tokens of the blog's twelve English posts as of 2026-09-29, and the tokens-per-word constant that minimizes squared error (the mean) and absolute error (a flat interval around the median). | CPU, instant |
| `step_size.py` | Gradient descent on the tokens-against-words line at several learning rates, on the raw feature and on the standardized one: where it converges, crawls and diverges. Imports the data from `a_loss.py`. | CPU, instant |
| `c_router.py` | The 24 hand-labeled reader questions, the serving-keyword count, and the logistic-regression router trained on it by full-batch gradient descent. | CPU, instant |
| `wrong_start.py` | The router started confidently wrong (w = -5) and trained for 500 steps with log-loss and with squared error. Imports from `c_router.py`. | CPU, instant |
| `minibatch.py` | The router trained with full-batch, mini-batch (4) and single-example updates, 200 shuffles each: updates to reach the optimum, and the final loss. Imports from `c_router.py`. | CPU, about 15 seconds |

The whole folder runs in about a minute on a CPU. The outputs here came from an Apple M4 laptop; the scripts need only numpy.

```bash
pip install "numpy==2.*"
python a_loss.py
python step_size.py
python c_router.py
python wrong_start.py
python minibatch.py
```
