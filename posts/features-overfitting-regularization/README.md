# Features and Overfitting: the full code

The scripts behind [Features and Overfitting: How a Model Memorizes, and How L1 and L2 Stop It](https://residual-stream.lvtan71.workers.dev/posts/features-overfitting-regularization/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `overfit.py` | The logistic-regression router on bag-of-words features, trained with no penalty for 25,600 steps on split 0: training loss, validation loss and the weight norm as it goes. Also defines `sigmoid`, `log_loss` and `train()` (with L2 or L1), which the other scripts import. | CPU, a few seconds |
| `sweep.py` | L2 and L1 at several strengths, each averaged over 20 random splits: training accuracy, validation loss and accuracy, nonzero weights. | CPU, about 20 seconds |
| `l1_words.py` | The words L1 keeps on split 0 at λ = 0.008, and their weights. | CPU, instant |
| `compare.py` | The hand-picked serving-keyword count against the bag of words, and both together, on the same 20 splits. | CPU, a few seconds |
| `honest.py` | Paired differences between λ settings over the 20 splits, with the naive and the Nadeau–Bengio corrected standard errors. | CPU, about 20 seconds |
| `peek.py` | Choosing λ on the test set instead of the validation set, and the test loss each choice reports. | CPU, about 10 seconds |

`router_data.py` is the helper the scripts import: it reads the questions, makes the stratified 30/10/10 split per route, and builds the bag-of-words features. `router-questions.csv` links to `../../data/router-questions.csv`: the 200 English and 20 Vietnamese reader questions written for this series, labeled by route.

The whole folder runs in about a minute on a CPU (the per-script times are estimates). The outputs here came from an Apple M4 laptop with numpy 2.4.4; the scripts need only numpy.

```bash
pip install "numpy==2.*"
python overfit.py
python sweep.py
python l1_words.py
python compare.py
python honest.py
python peek.py
```
