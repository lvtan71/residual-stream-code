import random
from lm import tok
from radix import RadixCache

# Cache-aware scheduling. 64 shared prefixes (2,048-token passages of this
# blog) with 16 different 128-token questions each, as in SGLang's
# shared-prefix benchmark; each answer is 256 tokens. All 1,024 requests are
# waiting at once, in random order. The cache holds 16,384 tokens, room for
# eight passages. Every round the scheduler admits 32 requests:
#   fcfs: in arrival order;
#   lpm:  the ones with the longest prefix already in the cache first
#         (SGLang's longest-prefix-match policy), matching every waiting
#         request against the tree each round;
#   lpm, as SGLang ships it: the same, but in arrival order while more than
#         128 requests wait, since matching a long queue costs time.
# Each round's batch is matched against the tree before it runs, and cached
# after. The cache is the room left beside the running requests. Hit rate =
# cached prompt tokens / prompt tokens. A request's wait is measured as the
# prompt tokens the server has computed by the time it is served.
def run(policy, seed, batch=32):
    rng = random.Random(seed)
    requests = []
    for g in range(GROUPS):
        passage = blog[g * P:(g + 1) * P]
        for k in range(PER):
            s = rng.randrange(GROUPS * P, len(blog) - Q)
            answer = [rng.randrange(151_000) for _ in range(A)]      # stands in for the generated tokens
            requests.append((passage + blog[s:s + Q], answer))
    rng.shuffle(requests)
    cache, waiting, matched, computed, wait = RadixCache(16_384), list(range(len(requests))), 0, 0, {}
    while waiting:
        if policy == "lpm" or (policy == "lpm, as shipped" and len(waiting) <= 128):
            match = {i: cache.match(requests[i][0]) for i in waiting}
            waiting.sort(key=lambda i: -match[i])                   # stable: ties stay in arrival order
        run_now, waiting = waiting[:batch], waiting[batch:]
        hits = {i: min(cache.match(requests[i][0]), len(requests[i][0]) - 1) for i in run_now}   # match the batch,
        for i in run_now:                                                                         # then run it
            prompt, answer = requests[i]
            matched += hits[i]
            computed += len(prompt) - hits[i]
            wait[i] = computed                  # prompt tokens computed by the time it is served
        for i in run_now:
            cache.insert(requests[i][0] + requests[i][1])
    total = sum(len(p) for p, _ in requests)
    early = sorted(wait[i] for i in range(128))                         # the first 128 to arrive
    return matched / total, early[len(early) // 2], early[-1]

blog = tok(open("blog_train.txt").read()).input_ids
P, Q, A, GROUPS, PER = 2048, 128, 256, 64, 16
print("                   prompt tokens reused       first 128 arrivals served after this much prefill work")
print("policy             seed 0  seed 1  seed 2     median (seed 0)     slowest (seed 0)")
for policy in ("fcfs", "lpm", "lpm, as shipped"):
    rows = [run(policy, seed) for seed in (0, 1, 2)]
    print(f"{policy:17s}  " + "  ".join(f"{r[0]:6.1%}" for r in rows) +
          f"     {rows[0][1]:9,d} tokens    {rows[0][2]:9,d} tokens")
