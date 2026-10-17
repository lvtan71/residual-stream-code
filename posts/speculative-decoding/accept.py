import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV = "cpu"   # the post's numbers come from a CPU run; another device can change the greedy answers

# How often guesses are accepted, measured. Target: Qwen2.5-1.5B-Instruct;
# draft: Qwen2.5-0.5B-Instruct would be the natural pair; this run uses the
# 0.5B base model, which has the same tokenizer. The target writes its answers (greedy, 120 tokens); then, at
# every position of each answer:
#   greedy acceptance: does the draft's top token equal the target's?
#   sampled acceptance at temperature 1: sum over tokens of min(p, q);
#   prompt lookup: take the last 2 tokens, find them earlier in prompt or
#   answer, and guess the token that followed; is it the target's token?
# Two kinds of task: rewriting a given paragraph (most of the answer can be
# copied from the prompt), and open questions (it can't).
T_ID, T_REV = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
D_ID, D_REV = "Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"
tok = AutoTokenizer.from_pretrained(T_ID, revision=T_REV)
target = AutoModelForCausalLM.from_pretrained(T_ID, revision=T_REV, dtype=torch.bfloat16).eval().to(DEV)
draft = AutoModelForCausalLM.from_pretrained(D_ID, revision=D_REV, dtype=torch.bfloat16).eval().to(DEV)
text = open("blog_val.txt").read()
paras = [p for p in text.split("\n\n") if 400 < len(p) < 900][:6]
tasks = {"rewrite a paragraph": ["Rewrite this paragraph in plain language, keeping its technical terms:\n\n" + p for p in paras],
         "open question": ["What is the KV cache, and why does it grow?", "Explain what a learning rate does.",
                           "Why do language models use tokenizers?", "What is attention in a Transformer?",
                           "How does dropout reduce overfitting?", "What is the difference between a GPU and a CPU?"]}

@torch.no_grad()
def measure(prompts):
    greedy, sampled, lookup, n = 0, 0.0, 0, 0
    for prompt in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)["input_ids"].to(DEV)
        out = target.generate(ids, max_new_tokens=120, do_sample=False, temperature=None, top_p=None, top_k=None,
                              repetition_penalty=1.0)
        full, start = out, ids.shape[1]
        p = target(full).logits[0, start - 1:-1].float().softmax(-1)
        q = draft(full).logits[0, start - 1:-1].float().softmax(-1)
        nxt = full[0, start:]
        greedy += (q.argmax(-1) == nxt).sum().item()
        sampled += torch.minimum(p, q).sum(-1).sum().item()
        seq = full[0].tolist()
        for i in range(start, len(seq)):
            a, b = seq[i - 2], seq[i - 1]
            guess = next((seq[j + 2] for j in range(i - 3, -1, -1) if seq[j] == a and seq[j + 1] == b), None)
            lookup += guess == seq[i]
        n += len(nxt)
    return greedy / n, sampled / n, lookup / n, n

for name, prompts in tasks.items():
    g, s, l, n = measure(prompts)
    print(f"{name:20s} ({n} tokens): draft model, greedy {g:.0%}, sampled {s:.0%}; prompt lookup {l:.0%}")