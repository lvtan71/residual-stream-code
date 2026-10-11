import json
import random
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from radix import RadixCache, BlockCache

# The three engines' prefix caches on one workload. SGLang matches any
# prefix to the token (a radix tree with one token per page); vLLM and
# TensorRT-LLM match whole blocks, 16 and 32 tokens by default. The caches
# differ in more than block size (eviction order, TensorRT-LLM's eviction
# priorities); here only the block size changes, with the eviction of
# radix.py. The workload is the one from the SGLang post:
# prefix reuse in real multi-turn chat. ShareGPT conversations with at least
# three user turns (up to eight), in Qwen2.5's chat format, starting over 40
# minutes. Each turn
# is a request whose prompt is the whole conversation so far; its answer is
# the reply the conversation actually got. Conversations start at random
# (one every 6 seconds on average) and a user replies 60 seconds after an
# answer on average, so turns of different conversations interleave. Each
# request is matched against the cache, then its prompt and answer are
# cached, as the engines do (requests are replayed one at a time, so an
# answer is cached before the next request arrives, and the cached answer is
# the reply ShareGPT recorded). No engine reuses a prompt's last token,
# since its logits are needed, and a block cache reuses whole blocks only.
# Reuse = cached prompt tokens / prompt tokens.
SHAREGPT = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json",
                           repo_type="dataset", revision="192ab2185289094fc556ec8ce5ce1e8e587154ca")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
rng = random.Random(0)
convs = [c["conversations"] for c in json.load(open(SHAREGPT))]
rng.shuffle(convs)

def msg(role, text):
    return tok(f"<|im_start|>{role}\n{text}<|im_end|>\n").input_ids

SYSTEM = msg("system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
ASSISTANT = tok("<|im_start|>assistant\n").input_ids
requests, t0, used = [], 0.0, 0
for c in convs:
    turns = [(c[i]["value"], c[i + 1]["value"]) for i in range(0, len(c) - 1, 2)
             if c[i]["from"] == "human" and c[i + 1]["from"] == "gpt"]
    if len(turns) < 3:
        continue
    t0 += rng.expovariate(1 / 6)
    used += 1
    t, history = t0, list(SYSTEM)
    for question, answer in turns[:8]:
        history += msg("user", question)
        prompt, reply = history + ASSISTANT, tok(f"{answer}<|im_end|>\n").input_ids
        if len(prompt) + len(reply) > 16_384:
            break
        requests.append((t, prompt, reply))
        history += ASSISTANT + reply
        t += rng.expovariate(1 / 60)
    if t0 > 400 * 6:
        break
requests.sort(key=lambda r: r[0])
prompt_tokens = sum(len(p) for _, p, _ in requests)
print(f"{len(requests)} requests from {used} multi-turn conversations; "
      f"{prompt_tokens:,} prompt tokens, {prompt_tokens / len(requests):.0f} per request")

print("prompt tokens reused, by cache size and match granularity (tokens)")
SIZES = (1, 16, 32, 64, 128)
print("cache size     " + "".join(f"{b:>8d}" for b in SIZES))
for cap in (65_536, 262_144, 10**9):
    row = []
    for b in SIZES:
        cache = RadixCache(cap) if b == 1 else BlockCache(cap, block=b)
        matched = 0
        for _, prompt, reply in requests:
            matched += min(cache.match(prompt), (len(prompt) - 1) // b * b)
            cache.insert(prompt + reply)
        row.append(matched / prompt_tokens)
    label = "unlimited" if cap == 10**9 else f"{cap:,}"
    print(f"{label:>10s}     " + "".join(f"{h:8.1%}" for h in row))
