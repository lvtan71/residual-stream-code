from harness import TASK, ORGANISMS, run_session, probe, save, print_usage

# 4. How often the default prompt keeps the plain corpus's identifiers:
#    compaction.py's plain/default condition, ten more runs.
compact = {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 50_000}}
kept_all = 0
for run in range(10):
    s = run_session(TASK, edits=[compact], max_tokens=16_000, label=f"plain corpus, default prompt, run {run + 1}")
    done = [e for e in s.events if e["kind"] == "compaction"]
    if not done:
        print("  no compaction")
        continue
    read = [n for slug, n, *_ in ORGANISMS
            if any(p == f"/research/{slug}_review.md" and t < done[0]["turn"] for t, p in s.reads)]
    kept = probe(s.summaries[0], read)
    ids = sum(kept[n]["obscure"] for n in read)
    kept_all += ids == len(read)
    print(f"  lifespans {sum(kept[n]['headline'] for n in read)}/{len(read)}, source identifiers {ids}/{len(read)}")
    save(f"repeat_{run + 1}", s, probe=kept)
print(f"summaries that kept every identifier: {kept_all} of 10")
print_usage()
