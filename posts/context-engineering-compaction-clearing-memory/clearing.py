from harness import TASK, run_session, save, print_usage

# 2. Clear old tool results once the context passes 30K tokens, keeping the
#    newest `keep`, and only when a pass frees at least 5K. Three settings of
#    keep for the six documents the task needs, three runs of each setting
#    that clears (keep 6 leaves nothing worth 5K to clear).
for keep in (2, 4, 6):
    clear = {"type": "clear_tool_uses_20250919",
             "trigger": {"type": "input_tokens", "value": 30_000},
             "keep": {"type": "tool_uses", "value": keep},
             "clear_at_least": {"type": "input_tokens", "value": 5_000},
             "exclude_tools": ["memory"]}
    for run in range(3 if keep < 6 else 1):
        s = run_session(TASK, edits=[clear], label=f"clearing, keep {keep}, run {run + 1}")
        paths = [p for _, p in s.reads]
        print(f"keep {keep}: {'finished' if s.finished else 'stopped at the turn limit'} after {len(s.trajectory)} turns; "
              f"peak context {s.peak:,}; {len(s.reads)} file reads, "
              f"{sum(paths.count(p) > 1 for p in set(paths))} documents read more than once")
        for e in s.events:
            print(f"  turn {e['turn']:2d}: cleared_tool_uses = {e['results']}, cleared_input_tokens = {e['tokens']:,}")
        save(f"clearing_keep{keep}_{run + 1}", s)
print_usage()
