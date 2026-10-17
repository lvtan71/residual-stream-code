# Memory traffic of one attention head's forward pass, in 16-bit numbers
# (2 bytes each), for sequence length n and head size d.
#
# Standard attention runs three separate kernels, and each one reads its
# inputs from GPU memory (HBM) and writes its result back:
#   S = Q K^T       reads Q, K (2nd), writes S (n^2)
#   P = softmax(S)  reads S, writes P (2 n^2)
#   O = P V         reads P, V (n^2 + nd), writes O (nd)
#
# FlashAttention, in the loop order FlashAttention-2 uses, keeps a block of
# B_r query rows in fast on-chip memory and streams K and V past it in
# blocks: Q and O cross HBM once, K and V once per block of query rows, and
# S and P never do. (This charges every re-read of K and V to HBM, a worst
# case: on a real GPU many of them hit the L2 cache.)
H100_SMEM = 228 * 1024                                    # bytes of shared memory per SM ("How a GPU Runs Deep Learning")

def standard(n, d):
    return 2 * (2 * n * d + n * n + 2 * n * n + n * n + n * d + n * d)

def flash(n, d, B_r):
    return 2 * (2 * n * d + (n // B_r) * 2 * n * d)       # Q and O once; K and V once per row block

d = 128
B_r = 128                                                 # 128 query rows of 128: 32 KB in 16 bits; fits with K, V blocks
                                                          # (the score tile and output accumulator live in registers)
print(f"one block of Q, K and V at {B_r} x {d} in 16 bits: {3 * B_r * d * 2 / 1024:.0f} KB of the {H100_SMEM // 1024} KB")
for n in (4096, 32768, 131072):
    s, f = standard(n, d), flash(n, d, B_r)
    print(f"n = {n:>7,}: standard {s / 2**20:9,.0f} MiB, flash {f / 2**20:7,.0f} MiB ({s / f:4.1f}x less); "
          f"the score matrix alone {n * n * 2 / 2**20:9,.0f} MiB per head")

# Bigger on-chip memory means bigger row blocks and fewer passes over K and V.
n = 32768
print(f"n = {n:,}: flash traffic with row blocks of " + ", ".join(
    f"{b}: {flash(n, d, b) / 2**20:,.0f} MiB" for b in (64, 128, 256)))

# What training keeps for the backward pass: standard attention saves P,
# n^2 numbers per head; FlashAttention saves only O and one number per row
# (the softmax's log-sum-exp) and recomputes the blocks of P when needed.
for n in (4096, 32768):
    print(f"n = {n:>6,}: saved for backward per head: standard {n * n * 2 / 2**20:7,.0f} MiB, "
          f"flash {(n * d + n) * 2 / 2**20:5.1f} MiB")