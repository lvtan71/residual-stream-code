# Small and Specialized Models: the full code

The scripts behind [Small and Specialized Models: When Not to Use a Frontier LLM](https://residual-stream.lvtan71.workers.dev/posts/small-and-specialized-models/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `bandwidth.py` | The bandwidth bound on batch-1 decoding: weight sizes of five models in 16 and 4 bits, the most tokens per second a laptop and an H100 could reach, and a check against Qwen2.5-1.5B's measured 41 ms per token. | Python, instant |
| `routing.py` | Four-way routing of the series' 200 English router questions, one out-of-sample prediction each: the bag-of-words router, MiniLM sentence embeddings with a softmax head and by nearest centroid, and Qwen2.5-1.5B-Instruct zero-shot and with 20 examples, read typed and free. Accuracy with bootstrap intervals, calibration, batch-1 time. | GPU, about 50 minutes; downloads Qwen2.5-1.5B-Instruct (3 GB) and all-MiniLM-L6-v2 (90 MB) at pinned revisions. Time it with nothing else using the GPU |

Helpers: `lm.py` loads Qwen2.5-1.5B-Instruct and its tokenizer. `router_data.py` (the questions and the 120 / 40 / 40 splits) is a copy of the helper from [Features and Overfitting](https://residual-stream.lvtan71.workers.dev/posts/features-overfitting-regularization/); `router4.py` (the four-way softmax router and its calibration error) and `softmax_basics.py` are copies of the helpers from [Cross-Entropy and the Softmax Gradient](https://residual-stream.lvtan71.workers.dev/posts/softmax-and-cross-entropy/). `quiet_machine.py` waits for the machine's load to drop before anything is timed; it is a reconstruction of a lost original and changes no result. `router-questions.csv` links to `../../data/router-questions.csv`, the series' hand-labeled router questions.

The outputs here came from an Apple M4 laptop with torch 2.14.0, transformers 5.17 and sentence-transformers 6.1.0. The huggingface_hub version was not recorded, so it is left unpinned.

```bash
pip install "numpy==2.*" "torch==2.14.*" "transformers==5.17.*" huggingface_hub "sentence-transformers==6.1.*"
python bandwidth.py
python routing.py
```
