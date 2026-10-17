import asyncio
import json
import sys
from mcp import Client, StdioServerParameters

# A host's side: launch the server as a subprocess, discover what it offers,
# and call a tool, all over JSON-RPC on the subprocess's stdin and stdout.
async def main():
    params = StdioServerParameters(command=sys.executable, args=["blog_server.py"])
    async with Client(params) as client:
        tools = await client.list_tools()
        for t in tools.tools:
            print(f"tool {t.name}: {t.description.splitlines()[0]}")
            print(f"  input schema: {json.dumps(t.input_schema)[:160]}...")
        print(f"protocol version: {client.protocol_version}")
        result = await client.call_tool("search_posts", {"query": "paged attention KV cache memory", "k": 3})
        print("search_posts ->", result.structured_content or result.content[0].text)
        text = await client.call_tool("read_post", {"slug": "kv-cache-compression", "max_words": 25})
        print("read_post ->", text.content[0].text)
        # The resource: the host, not the model, decides to read it and attach it.
        for t in (await client.list_resource_templates()).resource_templates:
            print(f"resource template {t.uri_template}: {t.description}")
        res = await client.read_resource("post://mixture-of-experts")
        print("resources/read ->", res.contents[0].text[:80], "...")
asyncio.run(main())
