import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# Supervised fine-tuning (SFT) on one chat example, through Qwen2.5's chat
# template. The model reads the whole conversation, but only the assistant's
# tokens are trained: every other position gets the label -100, which
# PyTorch's cross_entropy skips (ignore_index=-100).
tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                                                      revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"))
chat = [{"role": "user", "content": "What does the KV cache store?"},
        {"role": "assistant", "content": "The keys and values of every earlier token, so they aren't recomputed."}]
prompt = tok.apply_chat_template(chat[:1], tokenize=False, add_generation_prompt=True)
full = tok.apply_chat_template(chat, tokenize=False)
print(full)
p_ids, f_ids = tok(prompt).input_ids, tok(full).input_ids
assert f_ids[:len(p_ids)] == p_ids                        # the prompt is a prefix of the full text
labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
print(f"{len(f_ids)} tokens: {len(p_ids)} of prompt (system, user, assistant header), masked; "
      f"{len(f_ids) - len(p_ids)} trained:")
print(repr(tok.decode(f_ids[len(p_ids):])))

# In the loss, input positions 0..n-2 predict targets 1..n-1, and targets
# of -100 drop out of the average: F.cross_entropy(logits, y, ignore_index=-100).
y = torch.tensor(labels[1:])
print(f"targets in the loss: {(y != -100).sum().item()} of {len(y)}")