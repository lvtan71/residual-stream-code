# Serving Metrics and Prefill–Decode Disaggregation: the full code

The scripts behind [Serving Metrics and Prefill–Decode Disaggregation](https://residual-stream.lvtan71.workers.dev/posts/serving-metrics-and-disaggregation/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `serving_sim.py` | A step-by-step simulation of serving Llama-3-8B shapes on H100s, each step's time from the roofline: Poisson arrivals, continuous batching with whole or chunked prefill on colocated GPUs, and prefill and decode on separate GPUs with the KV cache sent over a link. Prints TTFT, TPOT and the worst gap between tokens, and the share of requests within two SLOs, on 2 and 4 GPUs. | CPU, a few seconds |

The outputs here came from an Apple M4 laptop; the script needs only numpy.

```bash
pip install "numpy==2.*"
python serving_sim.py
```
