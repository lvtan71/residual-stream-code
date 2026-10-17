import numpy as np

# A 3x3 output-stationary systolic array computing C = A @ B, cycle by cycle.
# Cell (i, j) owns C[i, j]. Row i of A enters from the left edge and moves
# one cell right per cycle; column j of B enters from the top and moves one
# cell down per cycle. Each input is skewed by one cycle per row (column),
# so A[i, k] and B[k, j] meet in cell (i, j) at cycle i + j + k. Every cell
# multiplies what arrives, adds to its own total, and passes both inputs on:
# no cell ever reads memory.
n = 3
A = np.arange(1, 10).reshape(n, n)
B = np.arange(1, 10).reshape(n, n)[::-1]
C = np.zeros((n, n), dtype=int)
a_reg = np.zeros((n, n), dtype=int)       # the A value each cell holds this cycle
b_reg = np.zeros((n, n), dtype=int)
macs = 0
for cycle in range(3 * n - 2):
    # shift: A moves right, B moves down; the edges take the next skewed input
    a_reg[:, 1:] = a_reg[:, :-1].copy(); b_reg[1:, :] = b_reg[:-1, :].copy()
    for i in range(n):
        k = cycle - i
        a_reg[i, 0] = A[i, k] if 0 <= k < n else 0
    for j in range(n):
        k = cycle - j
        b_reg[0, j] = B[k, j] if 0 <= k < n else 0
    C += a_reg * b_reg
    active = int(((a_reg != 0) & (b_reg != 0)).sum())
    macs += active
    print(f"cycle {cycle}: {active} of {n * n} cells multiplying")
print(f"\nresult equals A @ B: {np.array_equal(C, A @ B)}   ({macs} multiply-adds in {3 * n - 2} cycles)")
print(C)

# Memory traffic: each of A's and B's n^2 values is read from memory once
# and then reused n times as it passes through the array.
for n in (3, 128, 256):
    print(f"n = {n:3d}: {n ** 3:>10,d} multiply-adds from {2 * n * n:>7,d} values read: "
          f"each value used {2 * n ** 3 // (2 * n * n):3d} times")