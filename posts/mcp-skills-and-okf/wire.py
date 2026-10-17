import json
import subprocess
import sys

# The same exchange without an SDK: JSON-RPC 2.0 messages, one per line, on
# the server's stdin and stdout. In the 2026-07-28 protocol every request is
# self-contained: its _meta carries the protocol version and the client's
# capabilities, so no handshake has to come first.
proc = subprocess.Popen([sys.executable, "blog_server.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, text=True)
meta = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "wire-demo", "version": "0.1"}}
request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": "search_posts", "arguments": {"query": "speculative decoding draft", "k": 2}, "_meta": meta}}
print("->", json.dumps(request))
proc.stdin.write(json.dumps(request) + "\n"); proc.stdin.flush()
reply = json.loads(proc.stdout.readline())
print("<-", json.dumps(reply)[:700])
proc.terminate()
