import harness
from harness import TASK, ORGANISMS, run_session, probe, count_tokens, save, print_usage

# 3. Compaction at the API's minimum trigger, with the default prompt and with
#    instructions that name the details to keep, on the plain corpus and on
#    the dense one, three runs each. Each summary is probed for the planted
#    facts of the documents it summarized. max_tokens is raised to 16,000
#    because the summarizing step writes within it, and the default 4,096 cut
#    off summaries that listed every appendix row.
compact = {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 50_000}}
guided = dict(compact, instructions=(
    "Summarize this research agent transcript. Preserve every quantitative lifespan figure "
    "with its source organism, AND every appendix provenance detail (cohort/jar/rack/plate "
    "identifiers). Note which documents have been read and which remain. Wrap the summary "
    "in <summary></summary>."))

for dense in (False, True):
    harness.CORPUS.clear()
    harness.CORPUS.update(harness.make_corpus(dense=dense))
    corpus = "dense" if dense else "plain"
    print(f"{corpus} corpus: {sum(count_tokens(c) for c in harness.CORPUS.values()):,} tokens")
    for prompt, edit in (("default", compact), ("guided", guided)):
        for run in range(3):
            s = run_session(TASK, edits=[edit], max_tokens=16_000,
                            label=f"compaction, {corpus} corpus, {prompt} prompt, run {run + 1}")
            done = [e for e in s.events if e["kind"] == "compaction"]
            extra = {}
            if not done:
                print(f"  no compaction; peak context {s.peak:,}")
            else:
                first = done[0]["turn"]       # results of reads issued before this turn were summarized
                read = [n for slug, n, *_ in ORGANISMS
                        if any(p == f"/research/{slug}_review.md" and t < first for t, p in s.reads)]
                kept = probe(s.summaries[0], read)
                size = count_tokens(s.summaries[0])
                print(f"  {done[0]['before']:,} tokens -> {size:,}-token summary ({done[0]['before'] / size:.0f}x, "
                      f"{done[0]['summary_out']:,} output tokens billed) "
                      f"at turn {first}, final context {s.trajectory[-1][1]:,}; from the summary alone: "
                      f"lifespans {sum(kept[n]['headline'] for n in read)}/{len(read)}, "
                      f"source identifiers {sum(kept[n]['obscure'] for n in read)}/{len(read)}")
                extra = {"probe": kept, "summary_tokens": size}
            save(f"compaction_{corpus}_{prompt}_{run + 1}", s, **extra)
print_usage()
