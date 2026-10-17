import asyncio
import sys

# A stand-in for a reverse proxy between the user and the server, the kind
# every deployment has (nginx, a load balancer, a CDN). This one buffers the
# worst way a hop can: it collects the upstream response and forwards it
# only in blocks of 8 KB, or when the upstream finishes.
# With --honor it skips buffering for responses that ask for that with the
# header nginx uses, X-Accel-Buffering: no.
HONOR = "--honor" in sys.argv
BLOCK = 8192

async def pipe_up(reader, writer):
    while data := await reader.read(65536):
        writer.write(data); await writer.drain()

async def handle(client_r, client_w):
    up_r, up_w = await asyncio.open_connection("127.0.0.1", 8765)
    asyncio.create_task(pipe_up(client_r, up_w))
    buffered, head, passthrough = b"", None, False
    while data := await up_r.read(65536):
        if head is None:
            head = data.split(b"\r\n\r\n", 1)[0]
            passthrough = HONOR and b"x-accel-buffering: no" in head.lower()
        if passthrough:
            client_w.write(data); await client_w.drain(); continue
        buffered += data
        while len(buffered) >= BLOCK:
            client_w.write(buffered[:BLOCK]); buffered = buffered[BLOCK:]; await client_w.drain()
    client_w.write(buffered); await client_w.drain(); client_w.close()

async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", 8766)
    print("proxy on 8766" + (", honoring X-Accel-Buffering" if HONOR else ""), flush=True)
    async with server:
        await server.serve_forever()

asyncio.run(main())