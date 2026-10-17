import re
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# One research question over this blog, "How do the inference posts trade
# memory bandwidth against compute?", needs six posts. Their real sizes
# (Qwen2.5's tokenizer, standing in for any model's):
tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                                                      revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"))
slugs = ["kv-cache-explained", "flashattention-explained", "serving-metrics-and-disaggregation",
         "kv-cache-compression", "speculative-decoding", "quantization-from-scratch"]
docs = []
for s in slugs:
    body = open(f"content/posts/{s}.mdx").read().split("---", 2)[-1]
    docs.append(len(tok(re.sub(r"^import .*$", "", body, flags=re.M)).input_ids))
print("posts to read:", ", ".join(f"{d:,}" for d in docs), f"(total {sum(docs):,} tokens)")

# A cost model of input tokens (output tokens, fewer but pricier, are left
# out). Each model call rereads its whole context. With prompt caching
# ("Prompt Caching"), a call reads what the previous call sent at 0.1x the price and
# writes only what's new at 1.25x. Assumed: a 2,000-token system prompt and
# task; each post read over STEPS calls; 300 tokens of reasoning per call;
# workers get a 200-token brief and return 600-token summaries; the lead
# writes the briefs in one call and, once all workers have returned in
# parallel, synthesizes in one more.
SYS, THINK, BRIEF, SUMMARY = 2000, 300, 200, 600
def agent(prefix, chunks):
    """(input tokens read, cost with caching in uncached-token equivalents,
    largest context, model calls) for one agent reading chunks in order."""
    read, cached, context, prev = 0, 0.0, prefix, 0
    for c in list(chunks) + [0]:                     # the last call writes the answer
        read += context
        cached += 0.1 * prev + 1.25 * (context - prev)
        prev = context
        context += c + THINK if c else 0
    return read, cached, prev, len(chunks) + 1

def compare(steps, explore):
    """explore: how much each worker reads relative to its share (searching
    finds overlapping and irrelevant material)."""
    single = agent(SYS, [d / steps for d in docs for _ in range(steps)])
    workers = [agent(SYS + BRIEF, [d * explore / steps] * steps) for d in docs]
    lead_plan = agent(SYS, [])                                          # one call: write the six briefs
    lead_synth = agent(SYS + len(docs) * (BRIEF + SUMMARY), [])         # one call: read the summaries, answer
    multi = [sum(x[i] for x in [lead_plan, lead_synth] + workers) for i in (0, 1)]
    seq_multi = 1 + max(w[3] for w in workers) + 1
    return single, multi, max(w[2] for w in workers + [lead_synth]), seq_multi

print(f"\n{'assumptions':38s} {'':10s} {'tokens read':>12s} {'cached cost':>12s} {'largest context':>16s} {'sequential calls':>17s}")
for steps, explore, label in ((4, 1.0, "each post in 4 reads, no extra search"), (1, 1.0, "each post in 1 read"),
                              (4, 2.0, "workers search twice as much")):
    single, multi, wmax, seq = compare(steps, explore)
    print(f"{label:38s} {'single':10s} {single[0]:>12,.0f} {single[1]:>12,.0f} {single[2]:>16,.0f} {single[3]:>17d}")
    print(f"{'':38s} {'lead + 6':10s} {multi[0]:>12,.0f} {multi[1]:>12,.0f} {wmax:>16,.0f} {seq:>17d}")
print(f"\nwhat reaches the lead: {len(docs) * SUMMARY:,} tokens of summaries for {sum(docs):,} tokens of posts "
      f"({len(docs) * SUMMARY / sum(docs):.1%})")