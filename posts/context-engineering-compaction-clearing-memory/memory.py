from harness import ORGANISMS, run_session, MemoryBackend, count_tokens, save, print_usage

# 4. Session 1 reads half the corpus and saves notes to /memories. Session 2
#    asks for the whole comparison twice: with an empty memory directory and
#    with session 1's notes.
memory = MemoryBackend()
half = [f"/research/{slug}_review.md" for slug, *_ in ORGANISMS[:3]]
s1 = run_session("Compare these model organisms for aging research: " + ", ".join(half) +
                 ". Read each one and save your comparative findings to /memories so a "
                 "follow-up session can build on them.", memory=memory, label="session 1")
notes = memory.files()
print(f"session 1: {len(s1.reads)} file reads, peak {s1.peak:,}; saved {len(notes)} file(s), "
      f"{sum(count_tokens(t) for t in notes.values()):,} tokens")
save("memory_s1", s1, memories=notes)

S2 = ("Extend the comparison to ALL organisms in /research/, then give a final synthesis. "
      "Avoid redoing work that is already recorded.")
for name, backend in (("without", MemoryBackend()), ("with", memory)):
    s2 = run_session(S2, memory=backend, label=f"session 2, {name} memory")
    print(f"session 2 {name} memory: {len(s2.reads)} file reads, {len(s2.trajectory)} turns, peak {s2.peak:,}")
    save(f"memory_s2_{name}", s2)
print_usage()
