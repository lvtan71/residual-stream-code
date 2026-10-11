"""The mechanisms without the API: clearing as a pass over the message list,
and the memory backend's path guard. Runs offline."""
from harness import CORPUS, MemoryBackend

def clear_old_tool_results(messages, keep=3, placeholder="[cleared to save context]"):
    """Replace the content of all but the newest `keep` tool_result blocks."""
    spots = [(i, j) for i, m in enumerate(messages) if isinstance(m["content"], list)
             for j, b in enumerate(m["content"]) if b.get("type") == "tool_result"]
    for i, j in spots[:len(spots) - keep]:
        messages[i]["content"][j]["content"] = placeholder
    return messages

# A transcript of four reads: each assistant turn calls read_file, each user
# turn returns the document.
paths = list(CORPUS)[:4]
messages = [{"role": "user", "content": "Compare the organisms."}]
for k, p in enumerate(paths):
    messages.append({"role": "assistant", "content": [
        {"type": "tool_use", "id": f"t{k}", "name": "read_file", "input": {"path": p}}]})
    messages.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": f"t{k}", "content": CORPUS[p]}]})

def size(ms):
    return sum(len(str(b.get("content", b.get("input", "")))) for m in ms
               for b in (m["content"] if isinstance(m["content"], list) else [{"content": m["content"]}]))

before = size(messages)
clear_old_tool_results(messages, keep=1)
print(f"clearing, keep=1: {before:,} -> {size(messages):,} characters")
print("tool_use records kept:", [b["input"]["path"] for m in messages if m["role"] == "assistant"
                                 for b in m["content"] if b["type"] == "tool_use"])

memory = MemoryBackend()
print(memory.handle({"command": "create", "path": "/memories/notes.md", "file_text": "mouse: 30 months"}))
print(memory.handle({"command": "create", "path": "/memories/../../escape.txt", "file_text": "x"}))
