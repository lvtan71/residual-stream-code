# Generating L = 512 tokens with an 8B-parameter model on one H100 (989
# TFLOPS dense BF16, 3.35 TB/s), batch 1, ignoring attention's own cost.
# A forward pass costs about 2N FLOPs per position it processes, and reads
# all 2N bytes of BF16 weights once whatever the number of positions.
#   Autoregressive: L passes of 1 position each (the KV cache keeps earlier
#   positions). Each pass is memory-bound.
#   Masked diffusion: S passes over all L positions, because every position
#   can change at every step and a bidirectional model can't cache them.
N, L = 8e9, 512
flops_rate, bw = 989e12, 3.35e12
def step_time(positions):          # the slower of reading the weights and doing the arithmetic
    return max(2 * N / bw, 2 * N * positions / flops_rate)
ar_flops, ar_time = 2 * N * L, L * step_time(1)
print(f"autoregressive: {L} steps, {ar_flops / 1e12:6.1f} TFLOP, at least {ar_time:5.2f} s")
print(f"{'diffusion steps S':>18s} {'tokens per step':>16s} {'TFLOP':>8s} {'x AR FLOPs':>11s} {'time at least':>14s}")
for S in (512, 256, 128, 64, 32, 16):
    f = 2 * N * L * S
    print(f"{S:18d} {L / S:16.0f} {f / 1e12:8.0f} {f / ar_flops:11.0f} {S * step_time(L):12.2f} s")