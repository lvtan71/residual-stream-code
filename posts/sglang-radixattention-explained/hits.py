import json
import random
from huggingface_hub import hf_hub_download
from lm import tok
from radix import RadixCache, BlockCache

# Prefix reuse in real multi-turn chat. 400 ShareGPT conversations with at
# least three user turns (up to eight), in Qwen2.5's chat format. Each turn
# is a request whose prompt is the whole conversation so far; its answer is
# the reply the conversation actually got. Conversations start at random
# (one every 6 seconds on average) and a user replies 60 seconds after an
# answer on average, so turns of different conversations interleave. Each
# request is matched against the cache, then its prompt and answer are
# cached, as both engines do. Hit rate = cached prompt tokens / prompt tokens.
SHAREGPT = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json",
                           repo_type="dataset", revision="192ab2185289094fc556ec8ce5ce1e8e587154ca")
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
        prompt, reply = history + ASSISTANT, msg("assistant", answer)
        if len(prompt) + len(reply) > 16_384:
            break
        requests.append((t, prompt, reply))
        history += reply
        t += rng.expovariate(1 / 60)
    if t0 > 400 * 6:
        break
requests.sort(key=lambda r: r[0])
prompt_tokens = sum(len(p) for _, p, _ in requests)
print(f"{len(requests)} requests from {used} multi-turn conversations; "
      f"{prompt_tokens:,} prompt tokens, {prompt_tokens / len(requests):.0f} per request")

print("cache size (tokens)    radix tree, any prefix    16-token blocks")
for cap in (16_384, 65_536, 262_144, 1_048_576, 10**9):
    hits = []
    for cache in (RadixCache(cap), BlockCache(cap)):
        matched = 0
        for _, prompt, reply in requests:
            matched += min(cache.match(prompt), len(prompt) - 1)    # the last prompt token is always recomputed
            cache.insert(prompt + reply)
        hits.append(matched / prompt_tokens)
    label = "unlimited" if cap == 10**9 else f"{cap:,}"
    print(f"{label:>18s}          {hits[0]:6.1%}                  {hits[1]:6.1%}")
