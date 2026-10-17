# How tiling changes a matmul's arithmetic intensity, counted as float32
# reads from main memory with no help from caches: an upper bound on traffic.
# Each output element takes n multiply-adds (2n FLOPs) over a row of A and a
# column of B: 2n numbers read, if nothing is reused.
ridge_here, ridge_h100 = 2878 / 93, 989e3 / 3350      # measured laptop GPU; H100 datasheet
print(f"ridge points: this GPU {ridge_here:.0f} FLOP/byte, H100 {ridge_h100:.0f} FLOP/byte\n")
print(f"{'C tile':>9s} {'reads per output':>17s} {'intensity':>13s} {'threads, one output each':>26s}")
for T in (1, 16, 32, 64, 128):
    # A T x T tile of outputs reads a T-row strip of A and a T-column strip of
    # B: 2nT numbers for T*T outputs, so each number read is used T times.
    reads_per_output = 2 / T                          # in units of n numbers
    intensity = 2 / (reads_per_output * 4)            # FLOPs per byte
    label = "naive" if T == 1 else f"{T} x {T}"
    print(f"{label:>9s} {reads_per_output:>12.3f} x n {intensity:>8.2f} FLOP/B {T * T:>18,d}")