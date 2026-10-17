import glob
import os
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# What a GraphRAG index of this blog costs, against a vector index.
# From the paper (Edge et al., 2024):
#   600-token chunks with 100 tokens of overlap;
#   four community levels, C0 (root) to C3; its two corpora had 1.61 and 1.42
#     community summaries per chunk over all levels (Table 2);
#   summaries averaged 570-780 tokens; global search reads one level's
#     summaries per query: 2.3-2.6% of the chunked source text at C0,
#     67-74% at C3.
# Assumed here (the paper doesn't state them):
#   1,500 tokens of extraction prompt and examples, 500 tokens extracted;
#   one gleaning round per chunk: a yes/no check, then one more extraction
#     (in the paper the extraction runs only if the check says entities
#     were missed);
#   4,000 tokens of input per community summary (the window is 8,000).
# Not counted: merging entity descriptions into summaries, claim extraction.
# So the index cost is an estimate, more likely low than high.
tok = AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                                                      revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"))
EXCLUDE = {"agentic-graph-and-vectorless-rag", "recursive-language-models"}
total = sum(len(tok(open(p).read().split("---", 2)[-1]).input_ids)
            for p in glob.glob("content/posts/*.mdx")
            if os.path.basename(p)[:-4] not in EXCLUDE)
chunks = -(-(total - 100) // 500)                     # 600-token windows advancing 500 at a time
PROMPT, OUT, SUMM_IN, SUMM_OUT = 1500, 500, 4000, 700
extract_calls = chunks * 3                            # extract, check, glean
extract_in = chunks * ((PROMPT + 600) + (PROMPT + 600 + OUT) + (PROMPT + 600 + OUT + 1))
extract_out = chunks * (OUT + 1 + OUT)
print(f"corpus: {total:,} tokens -> {chunks:,} chunks of 600 tokens")
print(f"vector index: {chunks:,} chunks embedded once, 0 LLM calls\n")
for ratio in (1.42, 1.61):
    summaries = round(chunks * ratio)
    tin, tout = extract_in + summaries * SUMM_IN, extract_out + summaries * SUMM_OUT
    print(f"GraphRAG at {ratio} summaries per chunk: {extract_calls:,} extraction calls + {summaries:,} "
          f"summaries = {extract_calls + summaries:,} LLM calls")
    print(f"  {tin / 1e6:.1f}M tokens in, {tout / 1e6:.1f}M out ({tin / total:.0f}x the corpus as input); "
          f"${tin / 1e6 * 0.10 + tout / 1e6 * 0.40:,.2f} at $0.10/$0.40 per M, "
          f"${tin / 1e6 * 3 + tout / 1e6 * 15:,.2f} at $3/$15 per M")
source = chunks * 600                                 # the chunked text, overlap included (Table 2's base)
print(f"\nper global-search query: {0.025 * source:,.0f} tokens of summaries at C0, "
      f"{0.70 * source:,.0f} at C3 (map calls of up to 8,000 tokens each, before prompts: "
      f"{-(-0.025 * source // 8000):.0f} and {-(-0.70 * source // 8000):.0f})")
print("per vector-search query: one embedding, then the top 5 chunks, about 3,000 tokens")