import json

# What one streamed piece of text, "Two", costs on the wire in different
# formats, as compact JSON. The Anthropic event is the documented example's
# shape; the OpenAI chunk has the fields of the example in OpenAI's cookbook
# (current chunks also carry a padding field, below).
text = "Two"
ours = b"data: " + json.dumps({"text": text}).encode() + b"\n\n"
anthropic = (b"event: content_block_delta\ndata: " + json.dumps(
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
    separators=(",", ":")).encode() + b"\n\n")
openai = b"data: " + json.dumps(
    {"id": "chatcmpl-9lMgfRSWPHcw51s6wxKT1YEO2CKpd", "object": "chat.completion.chunk", "created": 1721075653,
     "model": "gpt-july-test", "system_fingerprint": "fp_e9b8ed65d2",
     "choices": [{"index": 0, "delta": {"content": text}, "logprobs": None, "finish_reason": None}]},
    separators=(",", ":")).encode() + b"\n\n"
payload = json.dumps({"text": text}).encode()
websocket = bytes([0x81, len(payload)]) + payload            # a server-to-client frame: 2 bytes of header
for name, b in (("this post's SSE event", ours), ("Anthropic content_block_delta", anthropic),
                ("OpenAI chat.completion.chunk", openai), ("this post's WebSocket frame", websocket)):
    print(f"{name:31s} {len(b):4d} bytes for {len(text.encode())} bytes of text")