# Streaming Tokens to the User: the full code

The scripts behind [Streaming Tokens to the User: SSE, WebSockets, and the Wait the Reader Feels](https://residual-stream.lvtan71.workers.dev/posts/streaming-tokens-sse-websockets/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `split_chars.py` | A Vietnamese sentence with two emoji, decoded one token at a time and incrementally: which pieces come out broken. | CPU, instant (tokenizer only). The first run downloads Qwen2.5-1.5B-Instruct (about 3 GB) at a pinned revision |
| `measure.py` | Starts `server.py` in the background and sends five prompts from Decoding, 100 tokens each, as one JSON response, as server-sent events and over a WebSocket: time to the first text and to the end, and bytes on the wire. | GPU, about 1.5 minutes once the model is downloaded |
| `event_bytes.py` | What one streamed piece of text costs on the wire as this post's SSE event, Anthropic's and OpenAI's event formats, and a WebSocket frame. | Python, instant |
| `reading_speed.py` | Tokens per word on the blog's prose (the 20 human continuations in `prompts.json`) against a reader's 238 words a minute, and what a reader waits for with and without streaming. | CPU, instant (tokenizer only) |
| `experiments.py` | Against a server it starts and restarts: an SSE client that leaves after 10 pieces (with the server watching its connection, then with `--naive`), a WebSocket client that says stop, and the same prompt through `proxy.py`, buffering and then with `--honor`. | GPU, about a minute |

Helpers, imported or started by the scripts above:
- `server.py`: the streaming server on `asyncio`, with `/complete`, `/stream` (SSE), `/ws` (a WebSocket written out by hand) and `/log` (what the server generated for each request, read by `experiments.py`), on port 8765. `python server.py [--naive]` runs it on its own.
- `client.py`: the measuring client on raw sockets.
- `proxy.py`: a reverse proxy on port 8766 that forwards only in 8 KB blocks, unless started with `--honor` and the response says `X-Accel-Buffering: no`.
- `launch.py`: starts the server or the proxy in the background and waits until it listens.
- `detokenizer.py`: incremental detokenizing, so a character split across tokens is sent whole.
- `lm.py`: loads Qwen2.5-1.5B-Instruct (the model from [Decoding](https://residual-stream.lvtan71.workers.dev/posts/decoding-from-logits-to-text/)), on the first GPU it finds: CUDA, then Apple's MPS, then the CPU.
- `sampling.py`: `choose` and the other decoding policies, a copy of the one in Decoding's folder.

`prompts.json` links to `../../data/prompts.json`: Decoding's 20 prompts, each about the first 200 characters of a paragraph from the blog, with the paragraph's real continuation.

The whole folder runs in about 3 minutes on the M4 once the model is downloaded; the per-script times are estimates. The times in the outputs depend on the machine. The outputs here came from an Apple M4 laptop with torch 2.14.0 and transformers 5.17.

```bash
pip install "torch==2.14.*" "transformers==5.17.*" huggingface_hub
python split_chars.py
python measure.py
python event_bytes.py
python reading_speed.py
python experiments.py
```
