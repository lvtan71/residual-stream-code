# LangChain and LangGraph, from the Inside: the full code

The scripts behind [LangChain and LangGraph, from the Inside: What the Frameworks Compile Down To](https://residual-stream.lvtan71.workers.dev/posts/langchain-and-langgraph-from-the-inside/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `lcel.py` | A prompt, a stub model and a parser composed with LangChain's pipe, and the same three steps called by hand. | CPU, instant |
| `agent_three_ways.py` | One calculator agent with a stub model, written by hand, as a LangGraph graph built from scratch, and with LangGraph's prebuilt tool node: the answers, the lines of code, and the time per run. Then a tools node that pauses for human approval, an edited tool call, the saved checkpoints, and a replay from one of them. | CPU, under a minute. The times depend on the machine |

No model and no API key: the model is a deterministic stub in both scripts. The outputs here came from an Apple M4 laptop with langchain-core 1.6 and LangGraph 1.2.

```bash
pip install "langchain-core==1.6.*" "langgraph==1.2.*"
python lcel.py
python agent_three_ways.py
```
