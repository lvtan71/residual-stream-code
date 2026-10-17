# Giving Agents Tools and Knowledge: the full code

The scripts behind [Giving Agents Tools and Knowledge: MCP, Skills and OKF](https://residual-stream.lvtan71.workers.dev/posts/mcp-skills-and-okf/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `blog_server.py` | An MCP server for the blog: a BM25 `search_posts` tool, a `read_post` tool and a `post://{slug}` resource, over stdio. Not run on its own: the next scripts launch it as a subprocess or import it. | |
| `mcp_client.py` | A host's side through the SDK: launches `blog_server.py`, lists its tools and resource template, calls both tools and reads a resource. | CPU, a few seconds |
| `wire.py` | The same tool call without an SDK: one JSON-RPC request written to the server's stdin, one reply read from its stdout. | CPU, a few seconds |
| `tool_cost.py` | Tokens that tool definitions and skill metadata take in every prompt, counted with Qwen2.5's tokenizer. | CPU, a few seconds; `snapshot_download` fetches the whole Qwen2.5-1.5B-Instruct repository (3 GB) at a pinned revision, though only the tokenizer is used |
| `okf.py` | Writes a small Open Knowledge Format bundle, `bundle/`, from the front matter of a few posts, and prints its index and one concept. | CPU, instant |

`blog-deb8375.tar.gz` links to `../../data/blog-deb8375.tar.gz`, a snapshot of the blog's posts at commit deb8375; `tar xzf` unpacks it into `content/`, which `blog_server.py` and `okf.py` read. The post's run read the blog's working tree with drafts in it, and this snapshot is the last commit before, so the BM25 scores in `mcp_client.out` and `wire.out` differ in the last digits from a rerun (5.28 against 5.75, for example); the ranking is the same.

The outputs here came from an Apple M4 laptop with mcp 2.2.0 and transformers 5.17.0 (torch 2.14.0).

`SKILL.md` is the example skill the post shows: front matter (name and description, the only part loaded up front) and the body the agent reads when the skill is used.

```bash
pip install "mcp==2.2.*" "transformers==5.17.*" huggingface_hub pyyaml
tar xzf blog-deb8375.tar.gz
python mcp_client.py
python wire.py
python tool_cost.py
python okf.py
```
