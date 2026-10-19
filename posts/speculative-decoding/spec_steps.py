import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV = "cpu"   # the post's numbers come from a CPU run; another device can change the greedy answers

# The same models, prompts and answers as accept.py, measured the way
# speculation actually runs:
#  - greedy speculation's tokens per step, exactly: with greedy decoding, a
#    run of accepted drafts leaves the draft with the same context as the
#    target, so per-position agreement on the target's text decides every step
#  - acceptance sum min(p, q) on text the target *sampled* at temperature 1
#  - prompt lookup as deployed: match the last 3, else 2, else 1 tokens
#    earlier in the text and propose up to 10 tokens that followed
torch.manual_seed(0)
T_ID, T_REV = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
D_ID, D_REV = "Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"
tok = AutoTokenizer.from_pretrained(T_ID, revision=T_REV)
target = AutoModelForCausalLM.from_pretrained(T_ID, revision=T_REV, dtype=torch.bfloat16).eval().to(DEV)
draft = AutoModelForCausalLM.from_pretrained(D_ID, revision=D_REV, dtype=torch.bfloat16).eval().to(DEV)
print("vocab", target.config.vocab_size, draft.config.vocab_size, flush=True)
text = open("blog_val.txt").read()
paras = [p for p in text.split("\n\n") if 400 < len(p) < 900][:6]
tasks = {"rewrite": ["Rewrite this paragraph in plain language, keeping its technical terms:\n\n" + p for p in paras],
         "open": ["What is the KV cache, and why does it grow?", "Explain what a learning rate does.",
                  "Why do language models use tokenizers?", "What is attention in a Transformer?",
                  "How does dropout reduce overfitting?", "What is the difference between a GPU and a CPU?"]}


def spec_steps_greedy(match, k):
    # match[i] = draft argmax == target token at answer position i
    i, steps, n = 0, 0, len(match)
    while i < n:
        r = 0
        while r < k and i + r < n and match[i + r]:
            r += 1
        i += r + 1
        steps += 1
    return n / steps


def lookup_propose(seq, i, max_ng=3, num=10):
    for ng in range(max_ng, 0, -1):
        if i - ng < 0:
            continue
        pat = seq[i - ng:i]
        for j in range(i - ng - 1, -1, -1):  # most recent earlier occurrence
            if seq[j:j + ng] == pat:
                return seq[j + ng:min(j + ng + num, i)]  # continuation from known prefix
    return []


def lookup_steps(seq, start, max_ng, num):
    i, steps, proposed_steps, hits_first = start, 0, 0, 0
    while i < len(seq):
        prop = lookup_propose(seq, i, max_ng, num)
        r = 0
        while r < len(prop) and i + r < len(seq) and prop[r] == seq[i + r]:
            r += 1
        if prop:
            proposed_steps += 1
            hits_first += r > 0
        i += r + 1
        steps += 1
    return (len(seq) - start) / steps, proposed_steps / steps, hits_first / max(proposed_steps, 1)


@torch.no_grad()
def measure(prompts):
    rows = []
    for prompt in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)["input_ids"].to(DEV)
        start = ids.shape[1]
        out = target.generate(ids, max_new_tokens=120, do_sample=False, temperature=None, top_p=None, top_k=None,
                              repetition_penalty=1.0)
        p = target(out).logits[0, start - 1:-1].float().softmax(-1)
        q = draft(out).logits[0, start - 1:-1].float().softmax(-1)
        nxt = out[0, start:]
        match = (q.argmax(-1) == nxt).tolist()
        smin_g = torch.minimum(p, q).sum(-1).mean().item()
        seq = out[0].tolist()
        # original single-token 2-gram metric
        lk = 0
        for i in range(start, len(seq)):
            a, b = seq[i - 2], seq[i - 1]
            guess = next((seq[j + 2] for j in range(i - 3, -1, -1) if seq[j] == a and seq[j + 1] == b), None)
            lk += guess == seq[i]
        # sampled T=1 text
        outs = target.generate(ids, max_new_tokens=120, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                               repetition_penalty=1.0)
        ps = target(outs).logits[0, start - 1:-1].float().softmax(-1)
        qs = draft(outs).logits[0, start - 1:-1].float().softmax(-1)
        smin_s = torch.minimum(ps, qs).sum(-1).mean().item()
        rows.append(dict(n=len(match), g=np.mean(match), smin_g=smin_g, smin_s=smin_s, ns=outs.shape[1] - start,
                         lk1=lk / len(match), spec4=spec_steps_greedy(match, 4), spec8=spec_steps_greedy(match, 8),
                         pl=lookup_steps(seq, start, 3, 10), pl2=lookup_steps(seq, start, 2, 10)))
        print({k: (np.round(v, 3) if not isinstance(v, tuple) else tuple(np.round(v, 3))) for k, v in rows[-1].items()}, flush=True)
    return rows


for name, prompts in tasks.items():
    rows = measure(prompts)
    N = sum(r["n"] for r in rows)
    w = lambda key: sum(r[key] * r["n"] for r in rows) / N
    g = w("g")
    print(f"== {name}: n={N}; greedy agree {g:.3f} (per-prompt min {min(r['g'] for r in rows):.2f} max {max(r['g'] for r in rows):.2f})")
    print(f"   smin on greedy text {w('smin_g'):.3f}; smin on T=1 sampled text {np.mean([r['smin_s'] for r in rows]):.3f}")
    print(f"   lookup 1-token 2-gram {w('lk1'):.3f}")
    print(f"   greedy spec tokens/step k=4: actual {np.mean([r['spec4'] for r in rows]):.2f}, iid formula {(1 - g**5)/(1 - g):.2f}; "
          f"k=8 actual {np.mean([r['spec8'] for r in rows]):.2f}, formula {(1 - g**9)/(1 - g):.2f}")
    for key in ("pl", "pl2"):
        print(f"   prompt lookup {key} (tokens/step, frac steps with a proposal, frac proposals with >=1 accepted): "
              f"{tuple(float(x) for x in np.round(np.mean([r[key] for r in rows], 0), 3))}", flush=True)