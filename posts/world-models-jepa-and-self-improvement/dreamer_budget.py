# What one Dreamer-style training update imagines. Batch shape and horizon
# follow DreamerV3's defaults: 16 sequences of 64 steps from the replay
# buffer, and 15 imagined steps from every one of those states. Imagination
# runs only the latent dynamics and the actor, critic and reward heads (no
# image encoder or decoder), batched across all start states at once.
B, T, H = 16, 64, 15
starts = B * T
imagined = starts * H
print(f"per update: {starts:,} start states from replay, {imagined:,} imagined steps,")
print(f"computed as {H} sequential batched steps of {starts:,} states each")
# Living through the same number of steps for real, one at a time:
for env_ms, what in ((1, "a fast simulator"), (50, "a robot acting at 20 Hz")):
    print(f"  {imagined:,} real steps in {what} ({env_ms} ms/step): {imagined * env_ms / 1000:,.0f} s")