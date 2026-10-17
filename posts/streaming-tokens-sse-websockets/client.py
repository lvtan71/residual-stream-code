import asyncio
import base64
import json
import os
import sys
import time

# Measuring the server from the outside, over raw sockets, so every byte on
# the wire is counted: when the first text arrives, when the last does, and
# how many bytes it took.
PORT = int(os.environ.get("PORT", 8765))

async def post(path, prompt, n, stop_after=None):
    """POST a prompt; returns (seconds to first text, seconds to the end,
    bytes received, pieces of text). Closes early after `stop_after` events."""
    r, w = await asyncio.open_connection("127.0.0.1", PORT)
    body = json.dumps({"prompt": prompt, "max_tokens": n}).encode()
    t0 = time.perf_counter()
    w.write(b"POST %s HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n" % (path.encode(), len(body)) + body)
    first, received, pieces, buffer = None, 0, [], b""
    head_done = False
    while chunk := await r.read(65536):
        received += len(chunk); buffer += chunk
        if not head_done and b"\r\n\r\n" in buffer:
            head_done, buffer = True, buffer.split(b"\r\n\r\n", 1)[1]
        if head_done and path == "/stream":
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                data = event[len(b"data: "):]
                if data != b"[DONE]":
                    first = first or time.perf_counter() - t0
                    pieces.append(json.loads(data)["text"])
            if stop_after and len(pieces) >= stop_after:
                w.close(); await asyncio.sleep(1.0)       # leave, then give the server a moment to notice
                return first, time.perf_counter() - t0, received, pieces
    if path == "/complete":
        first = time.perf_counter() - t0
        pieces = [json.loads(buffer)["text"]]
    return first, time.perf_counter() - t0, received, pieces

def ws_frame(payload):
    """A client-to-server text frame, masked as RFC 6455 requires."""
    mask, n = os.urandom(4), len(payload)
    size = bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + n.to_bytes(2, "big")
    return bytes([0x81]) + size + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

async def websocket(prompt, n, stop_after=None):
    r, w = await asyncio.open_connection("127.0.0.1", PORT)
    key = base64.b64encode(os.urandom(16))
    t0 = time.perf_counter()
    w.write(b"GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            b"Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: " + key + b"\r\n\r\n")
    received = len(await r.readuntil(b"\r\n\r\n"))
    w.write(ws_frame(json.dumps({"prompt": prompt, "max_tokens": n}).encode()))
    first, pieces = None, []
    while True:
        b0, b1 = await r.readexactly(2)
        size = b1 if b1 < 126 else int.from_bytes(await r.readexactly(2), "big")
        message = json.loads(await r.readexactly(size))
        received += 2 + (2 if b1 >= 126 else 0) + size
        if message.get("done"):
            break
        first = first or time.perf_counter() - t0
        pieces.append(message["text"])
        if stop_after and len(pieces) == stop_after:
            w.write(ws_frame(b'{"stop": true}'))           # same connection, other direction
    w.close()
    return first, time.perf_counter() - t0, received, pieces