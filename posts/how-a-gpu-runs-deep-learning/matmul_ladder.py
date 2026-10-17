import torch
from gpu_util import lib, timed_queued as timed

for n in (512, 1024, 2048):
    A, B = torch.randn(n, n, device="mps"), torch.randn(n, n, device="mps")
    exact = A.cpu().double() @ B.cpu().double()        # the reference, in float64 on the CPU
    row, worst = [f"n = {n:4d}:"], 0.0
    for name, run in [("naive", lambda C: lib.matmul_naive(A, B, C, n, threads=(n, n), group_size=(16, 16))),
                      ("tiled", lambda C: lib.matmul_tiled(A, B, C, n, threads=(n, n), group_size=(16, 16))),
                      ("tiled 32", lambda C: lib.matmul_tiled32(A, B, C, n, threads=(n, n), group_size=(32, 32))),
                      ("registers", lambda C: lib.matmul_registers(A, B, C, n, threads=(n // 4, n // 4),
                                                                   group_size=(16, 16))),
                      ("PyTorch", lambda C: torch.mm(A, B, out=C))]:
        C = torch.empty(n, n, device="mps")
        t = timed(lambda: run(C))
        worst = max(worst, (C.cpu().double() - exact).abs().max().item() / exact.abs().max().item())
        row.append(f"{name} {2 * n ** 3 / t / 1e9:5.0f} GFLOP/s")
    print("   ".join(row) + f"   (all within {worst:.0e} of float64)")