# A cascaded voice agent's response time, stage by stage: from the moment
# the user stops speaking to the moment they hear the first syllable of the
# reply. Every figure is an assumption for a well-built system, stated so it
# can be changed, not a measurement.
stages = [  # (stage, milliseconds, why)
    ("audio frame + jitter buffer", 40, "20 ms Opus frames; a jitter buffer of about one frame"),
    ("network, user -> server",     30, "one way, same region, WebRTC over UDP"),
    ("end-of-turn detection",      400, "VAD waits for this much silence before deciding the user is done"),
    ("speech-to-text, final",      100, "a streaming recognizer has the words already; it finalizes"),
    ("LLM time to first token",    250, "prefill of a cached 2,000-token prompt plus queueing"),
    ("LLM, first clause",          150, "TTS needs a few words to start: ~8 tokens at ~19 ms each"),
    ("text-to-speech, first audio", 120, "streaming TTS, time to its first audio chunk"),
    ("network, server -> user",     30, "one way"),
    ("playback buffer",             40, "the client's own jitter buffer before audio plays"),
]
total = sum(ms for _, ms, _ in stages)
print(f"{'stage':30s} {'ms':>5s} {'cumulative':>11s}")
run = 0
for name, ms, _ in stages:
    run += ms
    print(f"{name:30s} {ms:5d} {run:11d}")
print(f"\ntotal {total} ms; end-of-turn detection alone is {400 / total:.0%} of it")

# Two levers, applied to the same budget.
for name, change in (("semantic turn detection (400 -> 150 ms of silence)", {"end-of-turn detection": 150}),
                     ("speech-to-speech model (STT + LLM + TTS -> 350 ms)", None)):
    if change:
        t = sum(change.get(n, ms) for n, ms, _ in stages)
    else:
        t = sum(ms for n, ms, _ in stages if not n.startswith(("speech-to-text", "LLM", "text-to-speech"))) + 350
    print(f"  with {name}: {t} ms")