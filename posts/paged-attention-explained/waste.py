import json
import math
import random
import pandas as pd
from huggingface_hub import hf_hub_download
from lm import tok

# How much of the KV cache holds real tokens? Request lengths from the two
# workloads the vLLM paper used, ShareGPT (real ChatGPT conversations) and
# Alpaca (instructions), tokenized with Qwen2.5's tokenizer and filtered as
# vLLM's benchmark does (prompt at most 1,024 tokens, prompt plus answer at
# most 2,048). Four ways to reserve memory, from the paper's comparison:
#   max:    reserve the model's whole 2,048-token window for every request;
#   pow2:   the prompt plus the answer's length rounded up to a power of
#           two (knows the answer's length to within 2x);
#   oracle: exactly the prompt plus the answer (knows it in advance);
#   paged:  16-token blocks, one more only when the last fills.
# Usage is averaged over every decode step of every request: at each step a
# request holds its prompt and the answer so far, and occupies its reservation.
# The last column: how many requests run at once in the 15.7K token slots the
# paper's OPT-13B had for its KV cache (12 GB on one 40 GB A100).
SHAREGPT = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json",
                           repo_type="dataset", revision="192ab2185289094fc556ec8ce5ce1e8e587154ca")   # 673 MB
ALPACA = hf_hub_download("tatsu-lab/alpaca", "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
                         repo_type="dataset", revision="dce01c9b08f87459cf36a430d809084718273017")
MAX, BLOCK, N = 2048, 16, 5000
rng = random.Random(0)

def sharegpt():
    convs = [c["conversations"] for c in json.load(open(SHAREGPT))
             if len(c["conversations"]) >= 2]
    rng.shuffle(convs)
    return ((c[0]["value"], c[1]["value"]) for c in convs)

def alpaca():
    rows = pd.read_parquet(ALPACA).sample(frac=1, random_state=0)
    return ((r.instruction + ("\n\n" + r.input if r.input else ""), r.output) for r in rows.itertuples())

def lengths(pairs):
    out = []
    for prompt, answer in pairs:
        p, a = len(tok(prompt).input_ids), len(tok(answer).input_ids)
        if 4 <= p <= 1024 and a >= 4 and p + a <= MAX:
            out.append((p, a))
        if len(out) == N:
            return out

reserve = {
    "max": lambda p, a, held: MAX,
    "pow2": lambda p, a, held: p + 2 ** math.ceil(math.log2(a)),
    "oracle": lambda p, a, held: p + a,
    "paged": lambda p, a, held: BLOCK * math.ceil(held / BLOCK),
}
for name, pairs in (("ShareGPT", sharegpt()), ("Alpaca", alpaca())):
    reqs = lengths(pairs)
    json.dump(reqs, open(f"{name.lower()}_lengths.json", "w"))
    p_mean = sum(p for p, a in reqs) / N
    a_mean = sum(a for p, a in reqs) / N
    print(f"{name}: {N} requests, prompt {p_mean:.0f} tokens on average, answer {a_mean:.0f}")
    for way, r in reserve.items():
        held = reserved = 0
        for p, a in reqs:
            for t in range(1, a + 1):
                held += p + t
                reserved += r(p, a, p + t)
        steps = sum(a for p, a in reqs)
        print(f"  {way:6s} real tokens {held / reserved:6.1%} of reserved slots; "
              f"{reserved / steps:6.0f} slots per running request, {int(15_700 // (reserved / steps)):4d} running at once")
