import time
import torch
import kernels

# Auto-tuning, as an engine build does it: for each shape the engine will
# see, time every version of a kernel and keep the fastest. The kernel is the
# fused RMSNorm over Qwen2.5-1.5B's 1,536-wide rows; its versions differ in
# threads per row and numbers per read. The shapes: one row (decoding one
# sequence), 16 rows (decoding 16), 256 (a prefill chunk) and 4,096 (a long
# prefill).
N, EPS, CALLS = 1536, 1e-6, 200
VERSIONS = [(t, v) for t in (32, 64, 128, 256, 512, 1024) for v in (2, 4)]

def timed(fn):
    t0 = time.perf_counter()
    for _ in range(CALLS):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / CALLS

torch.manual_seed(0)
w = torch.randn(N).half().to("mps")
xs = {rows: torch.randn(rows, N).half().to("mps") for rows in (1, 16, 256, 4096)}
ys = {rows: torch.empty_like(x) for rows, x in xs.items()}
a = torch.randn(4096, 4096).half().to("mps")
t0 = time.perf_counter()
while time.perf_counter() - t0 < 3:           # bring the GPU's clock up first
    a @ a; torch.mps.synchronize()
runs = {(r, tv): [] for r in xs for tv in VERSIONS}
for round in range(9):                        # every shape and version once per round;
    for r, tv in runs:                        # the median over rounds is kept
        runs[r, tv].append(timed(lambda: kernels.rmsnorm(xs[r], w, EPS, *tv, out=ys[r])))
table = {r: {tv: sorted(runs[r, tv])[4] for tv in VERSIONS} for r in xs}
best = {r: min(table[r], key=table[r].get) for r in table}

print("microseconds per call; rows: 1, 16, 256, 4096 (fastest marked *)")
for tv in VERSIONS:
    cells = [f"{table[r][tv] * 1e6:6.1f}{'*' if best[r] == tv else ' '}" for r in table]
    print(f"{tv[0]:4d} threads x {tv[1]}  " + "  ".join(cells))
for r in table:
    fast, ties = table[r][best[r]], [tv for tv in VERSIONS if table[r][tv] < 1.05 * table[r][best[r]]]
    print(f"{r:4d} rows: fastest {fast * 1e6:5.1f} us, slowest {max(table[r].values()) / fast:.1f}x as long; "
          f"within 5% of the fastest: " + ", ".join(f"{t}x{v}" for t, v in ties))
