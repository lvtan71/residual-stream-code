# Knowledge Distillation: the full code

The scripts behind [Knowledge Distillation: Learning from a Teacher's Distribution](https://residual-stream.lvtan71.workers.dev/posts/knowledge-distillation/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `common.py` | Shared by the model scripts: downloads Qwen2.5-0.5B (the student), Qwen2.5-1.5B-Instruct (the teacher) and WikiText-2 at pinned revisions, tokenizes the blog text, and defines the training batches, bits per byte, and the KL to the teacher on held-out text. A helper, no output. | Downloads about 4 GB |
| `dark_knowledge.py` | The teacher's next-token probabilities after one sentence of the blog, at temperature 1 and 2. | GPU, under a minute |
| `tau_squared.py` | The soft loss's gradient on ten classes at temperatures 1 to 16, with and without the τ² factor, against the large-τ formula. | numpy, instant |
| `forward_reverse_kl.py` | One Gaussian fitted to a two-bump teacher by forward and by reverse KL, on a grid. | numpy, a few seconds |
| `teacher.py` | The teacher reads one seed's 400 training batches and saves its top-64 log-probabilities (`teacher_s<seed>.pt`); for seed 0 it also prints its own bits per byte. Run by `three_labels.py`. | GPU |
| `distill.py` | The student trained for 400 steps on hard labels, soft labels or their average (arguments: labels, learning rate, seed, and optionally `trace`), with bits per byte at five checkpoints. Run by the scripts below. | GPU, 16 GB of memory |
| `three_labels.py` | Runs `teacher.py` for seeds 0 and 1 and `distill.py` for the three kinds of label and two seeds, keeping each run's summary line in `teacher.log` and `distill.log`, then prints the table with `distill_table.py`. | GPU, the bulk of the folder: six training runs |
| `kl_trace.py` | Runs `teacher_val.py` (the teacher's top 64 on the held-out blog, `teacher_val.pt`), then `distill.py` for seed 0 of each kind of label with the KL to the teacher logged (`distill_trace.log`), then prints the table with `trace_table.py`. | GPU, three training runs |
| `soft_check_both.py` | Runs `soft_check.py` at learning rates 1e-5 and 1e-4: 100 steps on soft labels, the training loss split into the KL over the teacher's top 64 and the student's mass outside them. | GPU, two short runs |

`blog_train.txt` and `blog_val.txt` link to `../../data/blog-2026-10-04/`, the text of the blog's posts as they stood on 2026-10-04, written by the data script of [Sequences Before Attention](https://residual-stream.lvtan71.workers.dev/posts/rnns-and-why-transformers-won/). `runlog.py` holds the helper the three runner scripts share: it runs a script as a separate process and appends its summary line to a log, and also prints it to stderr to follow progress.

The nine 400-step training runs take several hours on a T4; per-script times are not recorded. The student's embedding table is frozen so that training fits in 16 GB. The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.18.0.

```bash
pip install "torch==2.14.*" "transformers==5.18.*" huggingface_hub pyarrow "numpy==2.*"
python dark_knowledge.py
python tau_squared.py
python forward_reverse_kl.py
python three_labels.py
python kl_trace.py
python soft_check_both.py
```
