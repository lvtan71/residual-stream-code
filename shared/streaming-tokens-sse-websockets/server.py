import asyncio
import base64
import hashlib
import json
import sys
import threading
import time
import torch
from lm import model, tok, DEVICE
from sampling import choose
from detokenizer import Detokenizer

# One model, three ways to deliver its tokens, on Python's standard library:
#   POST /complete  the whole answer as one JSON response, at the end;
#   POST /stream    server-sent events, one event per piece of text;
#   GET  /ws        a WebSocket, which also lets the client say "stop";
#   GET  /log       what was generated for each request so far.
# Generation runs in a thread and checks a stop flag before every token.
# With --naive the server never notices that a client has left.
NAIVE = "--naive" in sys.argv
LOG = []                                                  # (endpoint, tokens generated, client left?)

def generate(prompt, n, out, stop, loop):
    """Sample up to n tokens (top-p 0.9, seeded), pushing text pieces to the
    asyncio queue `out`; a None marks the end."""
    m, ids = model(), torch.tensor([tok(prompt).input_ids], device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    d, history, produced = Detokenizer(), [], 0
    with torch.no_grad():
        result = m(ids, use_cache=True)
        for _ in range(n):
            if stop.is_set():
                break
            token = choose(result.logits[0, -1], history, g, p=0.9)
            history.append(token); produced += 1
            if token == tok.eos_token_id:
                break
            loop.call_soon_threadsafe(out.put_nowait, d.push(token))
            result = m(torch.tensor([[token]], device=DEVICE), past_key_values=result.past_key_values, use_cache=True)
    loop.call_soon_threadsafe(out.put_nowait, None)
    return produced

async def start(prompt, n):
    out, stop, loop = asyncio.Queue(), threading.Event(), asyncio.get_running_loop()
    job = loop.run_in_executor(None, generate, prompt, n, out, stop, loop)
    return out, stop, job

async def handle(reader, writer):
    head = (await reader.readuntil(b"\r\n\r\n")).decode()
    method, path, _ = head.split("\r\n")[0].split(" ")
    headers = {k.lower(): v for k, v in (l.split(": ", 1) for l in head.split("\r\n")[1:] if ": " in l)}
    if path == "/ws":
        return await websocket(reader, writer, headers)
    if path == "/log":                                    # what the server did, for the experiments
        payload = json.dumps(LOG).encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(payload) + payload)
        return writer.close()
    body = json.loads(await reader.readexactly(int(headers["content-length"])))
    out, stop, job = await start(body["prompt"], body.get("max_tokens", 100))
    left = False
    if path == "/complete":
        pieces = []
        while (piece := await out.get()) is not None:
            pieces.append(piece)
        payload = json.dumps({"text": "".join(pieces)}).encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                     b"Content-Length: %d\r\nConnection: close\r\n\r\n" % len(payload) + payload)
    else:
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                     b"X-Accel-Buffering: no\r\nConnection: close\r\n\r\n")
        if not NAIVE:                                     # the client closing its side means it's gone
            asyncio.create_task(watch(reader, stop))
        while (piece := await out.get()) is not None:
            if piece:
                writer.write(b"data: " + json.dumps({"text": piece}).encode() + b"\n\n")
                try:
                    await writer.drain()
                except ConnectionError:
                    left = True
                    if not NAIVE:
                        stop.set()
        writer.write(b"data: [DONE]\n\n")
        left = left or stop.is_set()
    LOG.append((path, await job, left))
    print(f"{path}: generated {LOG[-1][1]} tokens, client left early: {left}", flush=True)
    try:
        await writer.drain(); writer.close()
    except ConnectionError:
        pass

async def watch(reader, stop):
    await reader.read()                                   # returns b"" when the client disconnects
    stop.set()

# --- WebSocket (RFC 6455), by hand: an HTTP upgrade, then framed messages.
async def websocket(reader, writer, headers):
    accept = base64.b64encode(hashlib.sha1((headers["sec-websocket-key"] +
                              "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
    writer.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                 b"Sec-WebSocket-Accept: " + accept + b"\r\n\r\n")
    request = json.loads(await read_frame(reader))
    out, stop, job = await start(request["prompt"], request.get("max_tokens", 100))
    async def listen():                                   # the client can talk while the server streams
        while True:
            message = await read_frame(reader)
            if message is None or json.loads(message).get("stop"):
                stop.set(); return
    listener = asyncio.create_task(listen())
    while (piece := await out.get()) is not None:
        if piece:
            writer.write(frame(json.dumps({"text": piece}).encode()))
    writer.write(frame(json.dumps({"done": True}).encode()))
    await writer.drain()
    LOG.append(("/ws", await job, stop.is_set()))
    print(f"/ws: generated {LOG[-1][1]} tokens, client said stop: {stop.is_set()}", flush=True)
    listener.cancel(); writer.close()

def frame(payload, opcode=0x1):
    """A server-to-client frame: FIN + opcode, then the length, then the data."""
    n = len(payload)
    head = bytes([0x80 | opcode]) + (bytes([n]) if n < 126 else bytes([126]) + n.to_bytes(2, "big"))
    return head + payload

async def read_frame(reader):
    """A client-to-server frame; clients must mask their data with 4 random bytes."""
    try:
        b0, b1 = await reader.readexactly(2)
    except asyncio.IncompleteReadError:
        return None
    n = b1 & 0x7F
    if n == 126:
        n = int.from_bytes(await reader.readexactly(2), "big")
    mask, data = await reader.readexactly(4), await reader.readexactly(n)
    if b0 & 0x0F == 0x8:                                  # a close frame
        return None
    return bytes(b ^ mask[i % 4] for i, b in enumerate(data)).decode()

async def main():
    model()                                               # load the weights before taking requests
    server = await asyncio.start_server(handle, "127.0.0.1", 8765)
    print("listening on 8765" + (" (naive)" if NAIVE else ""), flush=True)
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
