# Recursive Language Models: the full code

The scripts behind [Recursive Language Models: The Prompt as an Environment](https://residual-stream.lvtan71.workers.dev/posts/recursive-language-models/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `rlm.py` | A miniature recursive language model on Qwen2.5-3B-Instruct: the corpus sits in a REPL as `context`, the model writes code to read it and can call `llm_query` on a piece. Ten checkable tasks on the blog's posts (five planted needles, five counts), against reading the first 8,000 tokens directly and a fixed program making the same kind of sub-calls. Saves every turn and sub-call to `rlm_transcripts.json`. | GPU, about 93 minutes; downloads Qwen2.5-3B-Instruct (about 6 GB) at a pinned revision. Reads `content/posts/` from the blog snapshot |

`blog-73b66ae.tar.gz` links to `../../data/blog-73b66ae.tar.gz`, the blog's source at commit `73b66ae`; `tar xzf` unpacks its posts into `content/posts/`. `common.py` (the `clean` function that turns a post into prose) is a copy of the helper from [Building RAG](https://residual-stream.lvtan71.workers.dev/posts/building-rag-chunking-hyde-and-caching/). The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.18.0.

```bash
pip install "torch==2.14.*" "transformers==5.18.*" huggingface_hub
tar xzf blog-73b66ae.tar.gz
python rlm.py
```
