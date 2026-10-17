# What 4-bit weights buy at decode, by batch size, for an 8B model on an
# H100 (3.35 TB/s; 989 TFLOP/s dense BF16). One decode step for a batch of
# B sequences reads every weight once and does 2 FLOPs per weight per
# sequence; its time is at least the larger of the two ("How a GPU Runs Deep Learning"'s roofline;
# KV-cache reads and dequantization ignored). INT4 weights with group
# scales take about 4.25 bits each.
N, BW, FLOPS = 8e9, 3.35e12, 989e12
print(f"{'batch':>6s} {'BF16 step':>10s} {'INT4 step':>10s} {'speedup':>8s}   bound by")
for B in (1, 8, 64, 128, 256, 1024):
    compute = 2 * N * B / FLOPS
    bf16, int4 = max(2 * N / BW, compute), max(4.25 / 8 * N / BW, compute)
    print(f"{B:6d} {bf16 * 1e3:8.2f} ms {int4 * 1e3:8.2f} ms {bf16 / int4:7.1f}x   "
          f"BF16 {'memory' if 2 * N / BW > compute else 'compute'}, INT4 {'memory' if 4.25 / 8 * N / BW > compute else 'compute'}")
print(f"compute-bound from a batch of {4.25 / 8 * N / BW * FLOPS / (2 * N):.0f} (INT4) and "
      f"{2 * N / BW * FLOPS / (2 * N):.0f} (BF16)")