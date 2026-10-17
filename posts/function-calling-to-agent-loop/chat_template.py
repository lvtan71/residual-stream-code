import json
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# What function calling looks like as text: one exchange rendered through
# Qwen2.5's chat template. The tool's schema goes into the system prompt;
# the model's call is JSON between <tool_call> tags; the application's
# result goes back as a tool message, which the template wraps in
# <tool_response> tags. (Search result abridged.)
tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                                                      revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"))
tools = [{"type": "function", "function": {"name": "search_posts", "description": "Search the blog.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
call = {"name": "search_posts", "arguments": {"query": "online softmax running maximum"}}
result = [{"post": "flashattention-explained", "heading": "The online softmax", "snippet": "Keep a running maximum m..."}]
messages = [{"role": "system", "content": "You answer questions about a technical blog, using its tools."},
            {"role": "user", "content": "How can a maximum and a running sum be updated incrementally?"},
            {"role": "assistant", "content": f"<tool_call>\n{json.dumps(call)}\n</tool_call>"},
            {"role": "tool", "content": json.dumps(result)}]
print(tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False))
