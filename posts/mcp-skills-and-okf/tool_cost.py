import asyncio
import json
from transformers import AutoTokenizer
from mcp import Client
import blog_server
from huggingface_hub import snapshot_download

# What tool and skill definitions cost in context, in tokens (Qwen2.5's
# tokenizer, as a stand-in for any model's). Every tool a host connects is
# described in the prompt on every call, used or not.
tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                                                      revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"))
n = lambda text: len(tok(text).input_ids)

async def tool_defs():
    async with Client(blog_server.server) as client:                 # in-process, same server as before
        return [t.model_dump(by_alias=True, exclude_none=True) for t in (await client.list_tools()).tools]
defs = asyncio.run(tool_defs())
per_tool = sum(n(json.dumps(d)) for d in defs) / len(defs)
print(f"this blog's 2 tools: {sum(n(json.dumps(d)) for d in defs)} tokens of definitions, {per_tool:.0f} per tool")
no_out = sum(n(json.dumps({k: v for k, v in d.items() if k != "outputSchema"})) for d in defs)
print(f"  of which output schemas: {sum(n(json.dumps(d)) for d in defs) - no_out} tokens")
print(f"5 servers x 15 tools at that size: {5 * 15 * per_tool:,.0f} tokens in every request's prompt")

meta = ("name: pdf-processing\ndescription: Extract text and tables from PDF files, fill forms, merge documents. "
        "Use when working with PDF files or when the user mentions PDFs, forms, or document extraction.")
print(f"\none skill's metadata (name and description): {n(meta)} tokens")
for skills in (10, 50, 200):
    print(f"{skills:4d} skills: metadata always loaded {skills * n(meta):7,d} tokens; "
          f"every body loaded up front at ~4,000 tokens each {skills * 4000:9,d} tokens")
