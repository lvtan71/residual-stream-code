import asyncio
import json
import client
from client import post, websocket
from launch import start

# Three experiments against the running server (and, for the last, through
# the proxy on port 8766):
#   leave:   the client goes away after 10 pieces of text. Does the server stop?
#   stop:    over a WebSocket, the client asks it to stop after 10 pieces.
#   proxy:   when does the first text arrive through a buffering proxy?
PROMPT = json.load(open("prompts.json"))[3]["prompt"]

async def server_log():
    r, w = await asyncio.open_connection("127.0.0.1", 8765)
    w.write(b"GET /log HTTP/1.1\r\nHost: x\r\n\r\n")
    return json.loads((await r.read()).split(b"\r\n\r\n", 1)[1])

async def main(which):
    if which == "leave":
        await post("/stream", PROMPT, 100, stop_after=10)
        await asyncio.sleep(5)                            # let a server that didn't notice finish
        path, generated, left = (await server_log())[-1]
        print(f"SSE client left after 10 pieces: the server generated {generated} tokens in all")
    if which == "stop":
        await websocket(PROMPT, 100, stop_after=10)
        path, generated, stopped = (await server_log())[-1]
        print(f"WebSocket client said stop after 10 pieces: the server generated {generated} tokens in all")
    if which == "proxy":
        client.PORT = 8766
        first, total, received, pieces = await post("/stream", PROMPT, 100)
        print(f"through the proxy: first text after {first:.2f} s, the end after {total:.2f} s, "
              f"{len(pieces)} pieces, {received:,} bytes")

# The post ran main() once per experiment, restarting the server and the
# proxy with the flags each one needs.
server = start("server.py", port=8765)
asyncio.run(main("leave"))
server.terminate(); server.wait()
server = start("server.py", "--naive", port=8765)       # a server that never notices the client left
asyncio.run(main("leave"))
server.terminate(); server.wait()
server = start("server.py", port=8765)
asyncio.run(main("stop"))
proxy = start("proxy.py", port=8766)
asyncio.run(main("proxy"))
proxy.terminate(); proxy.wait()
proxy = start("proxy.py", "--honor", port=8766)        # honors X-Accel-Buffering: no
asyncio.run(main("proxy"))
proxy.terminate(); server.terminate()