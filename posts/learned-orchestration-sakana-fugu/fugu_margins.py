# Table 1 of the Sakana Fugu technical report (arXiv 2606.21228v2), as
# reported: Fugu-Ultra, Fugu, then the three workers both orchestrate.
# Baselines are provider-reported where available, so the comparison is
# not a controlled one. Margin = orchestrator minus the best single worker.
rows = {
    "SWE Bench Pro":          (73.7, 59.0, 69.2, 54.2, 58.6),
    "Terminal Bench 2.1":     (82.1, 80.2, 74.6, 70.3, 78.2),
    "LiveCodeBench":          (93.2, 92.9, 87.8, 88.5, 85.3),
    "LiveCodeBench Pro":      (90.8, 87.8, 84.8, 82.9, 88.4),
    "Humanity's Last Exam":   (50.0, 47.2, 49.8, 44.4, 41.4),
    "CharXiv Reasoning":      (86.6, 85.1, 84.2, 83.3, 84.1),
    "GPQA Diamond":           (95.5, 95.5, 92.0, 94.3, 93.6),
    "SciCode":                (58.7, 60.1, 53.5, 58.9, 56.1),
    "tau^3 Banking":          (20.6, 21.7, 20.6, 8.4, 20.6),
    "Long Context Reasoning": (73.3, 74.7, 67.7, 72.7, 74.3),
    "MRCRv2":                 (93.6, 86.6, 87.9, 84.9, 94.8),
}
workers = ("Claude Opus 4.8", "Gemini 3.1", "GPT-5.5")
print(f"{'benchmark':24s} {'best worker':>22s} {'Fugu-Ultra':>11s} {'Fugu':>7s}")
ahead = {"Fugu-Ultra": 0, "Fugu": 0}
for name, (ultra, fugu, *w) in rows.items():
    best = max(w)
    who = workers[w.index(best)]
    print(f"{name:24s} {who:>15s} {best:5.1f}  {ultra - best:+10.1f} {fugu - best:+7.1f}")
    ahead["Fugu-Ultra"] += ultra > best; ahead["Fugu"] += fugu > best
print(f"\nahead of the best single worker: Fugu-Ultra on {ahead['Fugu-Ultra']} of {len(rows)}, Fugu on {ahead['Fugu']} of {len(rows)}")
best_on = {}
for name, (_, _, *w) in rows.items():
    best_on.setdefault(workers[w.index(max(w))], []).append(name)
for k, v in best_on.items():
    print(f"  {k} is the best worker on {len(v)}: {', '.join(v)}")