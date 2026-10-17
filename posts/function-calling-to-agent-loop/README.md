# From Function Calling to the Agent Loop: the full code

The scripts behind [From Function Calling to the Agent Loop](https://residual-stream.lvtan71.workers.dev/posts/function-calling-to-agent-loop/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `chat_template.py` | One tool-calling exchange rendered through Qwen2.5's chat template: the tool schema in the system prompt, the call between `<tool_call>` tags, the result in `<tool_response>` tags. | CPU, seconds; downloads Qwen2.5-1.5B-Instruct (about 3 GB, only its tokenizer is used) at a pinned revision |
| `transcript_cost.py` | Tokens an agent's transcript reads over 5 to 50 turns, and the input cost with and without prompt caching. | Python, instant |
| `agent.py` | Qwen2.5 (1.5B or 3B, the argument) answering the 30 questions in `questions.json` four ways: one retrieval and one answer, and an agent loop with `search_posts` and `read_section` as is, told to search first, and with its first call forced. Search is BM25 plus MiniLM, fused by rank. Appends every answer to `agent_log_<size>.jsonl`. | GPU; downloads the model (3 GB for 1.5B, 6 GB for 3B) and all-MiniLM-L6-v2 at pinned revisions |
| `both_models.py` | Runs `agent.py 1.5B`, then `agent.py 3B`. | GPU, about 19 minutes |
| `rescore.py` | Rescores the logged answers: a citation counts only when the source line names exactly one of the blog's posts, the right one. Reads `agent_log_1.5B.jsonl` and `agent_log_3B.jsonl`. | Python, instant |

`questions.json` links to `../../data/questions.json`, the 30 questions written for [Beyond Vector RAG](https://residual-stream.lvtan71.workers.dev/posts/agentic-graph-and-vectorless-rag/). `blog-a7da2a2.tar.gz` links to `../../data/blog-a7da2a2.tar.gz`, the blog's posts at commit a7da2a2, which `agent.py` searches and `rescore.py` takes the post names from. `agent.py` appends to its logs, so delete `agent_log_*.jsonl` before running it again.

The whole folder takes about 19 minutes on an Apple M4 laptop GPU. The outputs here came from an Apple M4 laptop with torch 2.14.1 and transformers 5.18.0.

```bash
pip install numpy "torch==2.14.*" "transformers==5.18.*" sentence-transformers huggingface_hub
tar xzf blog-a7da2a2.tar.gz
python chat_template.py
python transcript_cost.py
python both_models.py
python rescore.py
```
