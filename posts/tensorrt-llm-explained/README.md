# TensorRT-LLM: the full code

The complete scripts behind [TensorRT-LLM: Compiling Inference Engines Ahead of Time](https://residual-stream.lvtan71.workers.dev/posts/tensorrt-llm-explained/). The post shows the important parts of each script, and each `.out` file is what the script printed for the post.

TensorRT-LLM itself runs only on Linux with NVIDIA GPUs. These scripts instead reproduce, on an Apple-silicon Mac, the two things its engine build did to a model: fusing operations into fewer kernels, and timing kernel versions per shape to keep the fastest. The kernels are written in Metal and run only on Apple silicon (through `torch.mps.compile_shader`).

Run them from this folder:

| Script | What it does | Needs |
| --- | --- | --- |
| `kernels.py` | Two fused Metal kernels: RMSNorm in one pass, and SiLU-and-multiply over a merged gate-and-up matmul. Imported by the next two. | |
| `tune.py` | Auto-tuning: times twelve versions of the RMSNorm kernel at four shapes. | Apple GPU, about a minute |
| `fuse.py` | Fuses Qwen2.5-1.5B-Instruct's RMSNorms and MLP activations; counts operations per decoded token, times decoding and prefill, compares outputs. | Apple GPU, about ten minutes. Reads `blog_train.txt`, a link to `../../data/blog_train.txt` |

`lm.py` loads Qwen2.5-1.5B-Instruct and its tokenizer. Time with nothing else using the GPU. The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.19.

```bash
pip install torch transformers huggingface_hub
python tune.py         # Apple-silicon Mac only (Metal)
python fuse.py         # Apple-silicon Mac only (Metal)
```
