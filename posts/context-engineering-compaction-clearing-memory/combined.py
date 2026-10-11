from harness import TASK, run_session, MemoryBackend, save, print_usage

# 5. All three: clearing early, compaction later, memory underneath and
#    excluded from clearing so the agent never clears its own notes.
memory = MemoryBackend()
s = run_session(TASK + " Save key findings to /memories as you go.", memory=memory, label="all three", edits=[
    {"type": "clear_tool_uses_20250919",
     "trigger": {"type": "input_tokens", "value": 30_000},
     "keep": {"type": "tool_uses", "value": 2},
     "exclude_tools": ["memory"]},
    {"type": "compact_20260112",
     "trigger": {"type": "input_tokens", "value": 50_000}}])

kinds = [e["kind"] for e in s.events]
print(f"peak context {s.peak:,} after {len(s.trajectory)} turns and {len(s.reads)} file reads; "
      f"{kinds.count('clearing')} clearing pass(es), {kinds.count('compaction')} compaction(s)")
save("combined", s, memories=memory.files())
print_usage()
