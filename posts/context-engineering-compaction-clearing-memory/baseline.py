from harness import TASK, CORPUS, run_session, count_tokens, save, print_usage

# 1. No context management: every file the agent reads stays in the window.
s = run_session(TASK, label="no management")

# What fills the window at its peak: the file reads, counted exactly.
read_tokens = sum(count_tokens(CORPUS[p]) for _, p in s.reads if p in CORPUS)
print(f"corpus: {len(CORPUS)} documents, {sum(count_tokens(c) for c in CORPUS.values()):,} tokens")
print(f"peak context {s.peak:,} tokens after {len(s.trajectory)} turns and {len(s.reads)} file reads")
print(f"  file-read results  {read_tokens:>7,}  {read_tokens / s.peak:6.1%}")
print(f"  everything else    {s.peak - read_tokens:>7,}  {1 - read_tokens / s.peak:6.1%}")
save("baseline", s, read_tokens=read_tokens)
print_usage()
