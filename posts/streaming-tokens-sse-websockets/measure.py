import asyncio
import json
from client import post, websocket
from launch import start

# Five prompts from the "Decoding" set, 100 tokens each, delivered three ways.
prompts = [p["prompt"] for p in json.load(open("prompts.json"))][:5]

async def main():
    rows = {"/complete": [], "/stream": [], "/ws": []}
    for prompt in prompts:
        rows["/complete"].append(await post("/complete", prompt, 100))
        rows["/stream"].append(await post("/stream", prompt, 100))
        rows["/ws"].append(await websocket(prompt, 100))
    for name, rs in rows.items():
        first = sum(r[0] for r in rs) / len(rs); total = sum(r[1] for r in rs) / len(rs)
        text = ["".join(r[3]) for r in rs]
        received = sum(r[2] for r in rs); chars = sum(len(t.encode()) for t in text)
        print(f"{name:10s} first text after {first:5.2f} s, all of it after {total:5.2f} s; "
              f"{received / 5:6.0f} bytes on the wire per answer, for {chars / 5:4.0f} bytes of text "
              f"({received / chars:.1f}x)")
    same = all("".join(a[3]) == "".join(b[3]) == "".join(c[3]) for a, b, c in zip(*rows.values()))
    print(f"the same text all three ways: {same}")

server = start("server.py", port=8765)
try:
    asyncio.run(main())
finally:
    server.terminate()
