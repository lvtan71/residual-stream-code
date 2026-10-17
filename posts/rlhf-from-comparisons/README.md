# RLHF: the full code

The scripts behind [RLHF: From Christiano 2017 to InstructGPT to Today's Recipe](https://residual-stream.lvtan71.workers.dev/posts/rlhf-from-comparisons/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `bradley_terry.py` | Bradley-Terry scores for three answers fitted by gradient ascent to 13 pairwise outcomes, and the win probabilities they imply against the observed ones. | CPU, instant |
| `overoptimization.py` | RLHF solved exactly on 512 three-token responses: a reward model fitted to 300 noisy comparisons from a hidden gold reward, then the KL-regularized optimal policy for eight values of beta, in 100 worlds of each of three kinds. Prints KL, proxy gain, gold gain and sevens per answer. | CPU, about 10 seconds |

The whole folder runs in well under a minute on a CPU. The outputs here came from an Apple M4 laptop; the scripts need only numpy.

```bash
pip install "numpy==2.*"
python bradley_terry.py
python overoptimization.py
```
