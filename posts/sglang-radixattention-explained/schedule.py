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
# Hit rate = cached prompt tokens / prompt tokens. The wait of a request is
# the round it is served in.
rng = random.Random(0)
blog = tok(open("blog_train.txt").read()).input_ids
P, Q, A, GROUPS, PER, BATCH = 2048, 128, 256, 64, 16, 32
requests = []
for g in range(GROUPS):
    passage = blog[g * P:(g + 1) * P]
    for k in range(PER):
        s = rng.randrange(GROUPS * P, len(blog) - Q)
        answer = [rng.randrange(151_000) for _ in range(A)]          # stands in for the generated tokens
        requests.append((passage + blog[s:s + Q], answer))
rng.shuffle(requests)

for policy in ("fcfs", "lpm", "lpm, as shipped"):
    cache, waiting, matched, rounds = RadixCache(16_384), list(range(len(requests))), 0, {}
    r = 0
    while waiting:
        if policy == "lpm" or (policy == "lpm, as shipped" and len(waiting) <= 128):
            match = {i: cache.match(requests[i][0]) for i in waiting}
            waiting.sort(key=lambda i: -match[i])                   # stable: ties stay in arrival order
        batch, waiting = waiting[:BATCH], waiting[BATCH:]
        for i in batch:
            prompt, answer = requests[i]
            matched += min(cache.match(prompt), len(prompt) - 1)
            cache.insert(prompt + answer)
            rounds[i] = r + 1                                       # rounds counted from 1
        r += 1
    total = sum(len(p) for p, _ in requests)
    first_rounds = [rounds[i] for i in range(BATCH * 4)]                # the first 128 to arrive
    print(f"{policy}: hit rate {matched / total:5.1%}, prompt tokens computed {total - matched:,} of {total:,}; "
          f"the first 128 arrivals all served by round {max(first_rounds)} of {r}")
