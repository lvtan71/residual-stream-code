# Measuring a Classifier Honestly: the full code

The scripts behind [Measuring a Classifier Honestly: Precision, Recall, Thresholds and Calibration](https://residual-stream.lvtan71.workers.dev/posts/measuring-a-classifier/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `router_data.py` | Helper: loads `router-questions.csv`, splits the English questions 30/10/10 per route, and builds the bag-of-words features. Prints nothing. | CPU |
| `overfit.py` | Helper: `sigmoid`, `log_loss` and the logistic-regression trainer with an L2 or L1 penalty. Run on its own it prints the overfitting run of its source post. | CPU |
| `router_models.py` | Trains the three routers (no penalty, L2, L2 + keyword count) on 20 random splits and keeps their logits on the train, validation, test and Vietnamese questions. Prints nothing; the other scripts import it. | CPU, a few seconds |
| `measure.py` | The confusion matrix of one split; accuracy, precision, recall, F1, ROC-AUC and average precision of the three routers over 800 pooled test predictions; the threshold sweep; three ways to pick a threshold when a miss costs twice a false alarm. | CPU, a few seconds |
| `calibration.py` | Reliability tables and ECE of two routers, the ECE floor of perfectly calibrated labels, and temperature scaling fit on each split's validation set. | CPU, a few seconds |
| `vietnamese.py` | The routers on 20 Vietnamese questions they never trained on: which words the English vocabulary sees, recall, false alarms and per-question catch rates over 20 splits. | CPU, a few seconds |

`router-questions.csv` links to `../../data/router-questions.csv`, the 200 English and 20 Vietnamese hand-labeled questions. `router_data.py` and `overfit.py` are copies of the helpers from the post [Features and Overfitting](https://residual-stream.lvtan71.workers.dev/posts/features-overfitting-regularization/).

The whole folder runs in about a minute on a CPU. The outputs here came from an Apple M4 laptop; the scripts need only numpy.

```bash
pip install "numpy==2.*"
python measure.py
python calibration.py
python vietnamese.py
```
