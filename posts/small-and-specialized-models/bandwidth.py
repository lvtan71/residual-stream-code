# What "small" buys. Decoding one token reads every weight once, so with a
# batch of one, generation speed is bounded by memory bandwidth: tokens per
# second <= bandwidth / bytes of weights ("How a GPU Runs Deep Learning"'s roofline, memory-bound).
# Bandwidths: this laptop's GPU as measured in "How a GPU Runs Deep Learning" (93 GB/s), and an
# H100's 3.35 TB/s. Memory: the laptop's 16 GB, shared with everything else,
# and the H100's 80 GB; "-" marks weights that don't fit. Reading the KV
# cache ("The KV Cache") is left out: small at short context, not at long context.
LAPTOP, H100 = (93e9, 16e9), (3.35e12, 80e9)
models = [("0.5B", 0.5e9), ("1.5B", 1.5e9), ("3B", 3e9), ("8B", 8e9), ("70B", 70e9)]

def cap(device, size):
    bandwidth, memory = device
    return f"{bandwidth / size:5.0f}" if size < memory else "    -"

print(f"{'':6s} {'BF16':>8s} {'INT4':>8s}   tokens/s at most, batch 1:  laptop BF16 / INT4    H100 BF16 / INT4")
for name, n in models:
    bf16, int4 = 2 * n, 0.5 * n
    print(f"{name:6s} {bf16 / 1e9:6.1f}GB {int4 / 1e9:6.2f}GB   {'':26s}  {cap(LAPTOP, bf16)} / {cap(LAPTOP, int4)}    "
          f"{cap(H100, bf16)} / {cap(H100, int4)}")
# A check against a measurement: "Decoding" timed Qwen2.5-1.5B (1.54B
# parameters, 16-bit) at 41 ms per token on this laptop.
n = 1.54e9
print(f"\nQwen2.5-1.5B in 16 bits: {2 * n / 1e9:.2f} GB per token read; at the measured 41 ms per token that's "
      f"{2 * n / 0.041 / 1e9:.0f} GB/s, {2 * n / 0.041 / LAPTOP[0]:.0%} of the measured bandwidth")