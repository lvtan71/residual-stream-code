import time
import torch
import torch.nn.functional as F

if not torch.backends.mps.is_available():
    raise SystemExit("kernel.py is a Metal kernel and runs only on an Apple-silicon Mac")

# Decode attention over a paged KV cache, as a Metal kernel: one query
# token per sequence attends over its S cached tokens, reading every key
# and value through the sequence's block table. Qwen2.5-1.5B's attention
# shapes: 12 query heads sharing 2 key-value heads, 128 dimensions each,
# 16-bit numbers. Laid out the way fast decode kernels are: each threadgroup
# takes one key-value head of one sequence and a 256-token slice of its
# cache, reads each key and value once for all 6 query heads that share
# them, and keeps a running softmax (as FlashAttention does); a second kernel merges the
# partial results.
# The pool is [blocks, kv heads, BLOCK, 128]. The baseline is a contiguous
# cache, [sequences, kv heads, S, 128], addressed directly with no table.
SEQS, NQ, NK, D, S = 16, 12, 2, 128, 4096
SRC = """
#include <metal_stdlib>
using namespace metal;
constant uint NQ = 12, NK = 2, G = NQ / NK, D = 128, S = CONTEXT, BLOCK = BLOCK_SIZE, NB = S / BLOCK;
constant uint SLICE = 256, NS = S / SLICE;          // tokens per threadgroup; slices per sequence

uint row(device const int* table, uint seq, uint h, uint t) {      // where token t's key or value starts
#if NO_TABLE                                         // the contiguous cache: [sequences, kv heads, S, D]
    return ((seq * NK + h) * S + t) * D;
#else
    uint b = table[seq * NB + t / BLOCK];
    return ((b * NK + h) * BLOCK + t % BLOCK) * D;
#endif
}

// threadgroup (slice, kv head, sequence); 4 SIMD groups of 32 threads, each
// taking every 4th token of the slice; lane j holds dimensions 4j to 4j+3.
kernel void attend(device const half* q, device const half* k, device const half* v,
                   device const int* table, device float* part_m, device float* part_l,
                   device float4* part_acc, uint3 tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint slice = tg.x, h = tg.y, seq = tg.z;
    float4 qv[G], acc[G];
    float m[G], l[G];
    for (uint g = 0; g < G; g++) {
        qv[g] = float4(((device const half4*)(q + (seq * NQ + h * G + g) * D))[lane]) * rsqrt(float(D));
        acc[g] = 0; m[g] = -INFINITY; l[g] = 0;
    }
    for (uint t = slice * SLICE + sg; t < (slice + 1) * SLICE; t += 4) {
        uint r = row(table, seq, h, t);
        float4 k4 = float4(((device const half4*)(k + r))[lane]);
        float4 v4 = float4(((device const half4*)(v + r))[lane]);
        for (uint g = 0; g < G; g++) {                       // running softmax, per query head
            float s = simd_sum(dot(qv[g], k4));
            float mn = max(m[g], s), a = exp(m[g] - mn), e = exp(s - mn);
            l[g] = l[g] * a + e; acc[g] = acc[g] * a + e * v4; m[g] = mn;
        }
    }
    uint p = (seq * NK + h) * NS * 4 + slice * 4 + sg;     // one partial result per SIMD group
    for (uint g = 0; g < G; g++) {
        uint i = p * G + g;
        if (lane == 0) { part_m[i] = m[g]; part_l[i] = l[g]; }
        part_acc[i * 32 + lane] = acc[g];
    }
}

// threadgroup (query head, sequence), one thread per dimension: merge the partials.
kernel void merge(device const float* part_m, device const float* part_l, device const float* part_acc,
                  device float* out, uint2 tg [[threadgroup_position_in_grid]],
                  uint2 tid [[thread_position_in_threadgroup]]) {
    uint qh = tg.x, seq = tg.y, d = tid.x, h = qh / G, g = qh % G, base = (seq * NK + h) * NS * 4;
    float M = -INFINITY;
    for (uint p = 0; p < NS * 4; p++) M = max(M, part_m[(base + p) * G + g]);
    float L = 0, o = 0;
    for (uint p = 0; p < NS * 4; p++) {
        uint i = (base + p) * G + g;
        float w = exp(part_m[i] - M);
        L += part_l[i] * w; o += part_acc[i * D + d] * w;
    }
    out[(seq * NQ + qh) * D + d] = o / L;
}

// Copying alone: every key and value out through the table, no arithmetic.
kernel void gather(device const half4* k, device const half4* v, device const int* table,
                   device half4* ko, device half4* vo, uint i [[thread_position_in_grid]]) {
    uint lane = i % 32, t = i / 32 % S, h = i / 32 / S % NK, seq = i / 32 / S / NK;
    uint r = row(table, seq, h, t) / 4 + lane;
    ko[i] = k[r]; vo[i] = v[r];
}
"""

def setup(block, scattered, gen, table=True):
    nb = S // block
    src = SRC.replace("CONTEXT", str(S)).replace("BLOCK_SIZE", str(block))
    lib = torch.mps.compile_shader(("" if table else "#define NO_TABLE 1\n") + src)
    order = torch.randperm(SEQS * nb, generator=gen) if scattered else torch.arange(SEQS * nb)
    table = order.view(SEQS, nb).int().to("mps")
    return lib, table

PARTS = SEQS * NK * (S // 256) * 4 * (NQ // NK)
part_m = torch.empty(PARTS, device="mps")
part_l = torch.empty(PARTS, device="mps")
part_acc = torch.empty(PARTS, D, device="mps")

def run(lib, table, q, kpool, vpool, out):
    lib.attend(q, kpool, vpool, table, part_m, part_l, part_acc,
               threads=(S // 256 * 128, NK, SEQS), group_size=(128, 1, 1))
    lib.merge(part_m, part_l, part_acc, out, threads=(NQ * D, SEQS), group_size=(D, 1))

def read(lib, table, kpool, vpool, ko, vo):
    lib.gather(kpool, vpool, table, ko, vo, threads=SEQS * NK * S * 32, group_size=256)

def pools(K, V, table, block):
    """Lay contiguous K, V [SEQS, NK, S, D] out in the pool as the table says."""
    nb = S // block
    kp = torch.empty(SEQS * nb, NK, block, D, dtype=K.dtype, device="mps")
    vp = torch.empty_like(kp)
    kp[table.long().flatten()] = K.view(SEQS, NK, nb, block, D).transpose(1, 2).reshape(-1, NK, block, D)
    vp[table.long().flatten()] = V.view(SEQS, NK, nb, block, D).transpose(1, 2).reshape(-1, NK, block, D)
    return kp, vp

def timed(fn, trials=50):
    fn(); torch.mps.synchronize()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter(); fn(); torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[trials // 2]

if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(SEQS, NQ, D, generator=gen).half().to("mps")
    K = torch.randn(SEQS, NK, S, D, generator=gen).half().to("mps")
    V = torch.randn(SEQS, NK, S, D, generator=gen).half().to("mps")
    ref = F.scaled_dot_product_attention(q.float()[:, :, None], K.float().repeat_interleave(NQ // NK, 1),
                                         V.float().repeat_interleave(NQ // NK, 1))[:, :, 0]
    out = torch.empty(SEQS, NQ, D, device="mps")
    moved = 2 * K.numel() * 2                                  # bytes of keys and values read
    ko, vo = torch.empty_like(K), torch.empty_like(V)
    lib, table = setup(S, False, gen, table=False)                # a block of S tokens, in order, is the
    kp, vp = pools(K, V, table, S)                               # contiguous layout; NO_TABLE skips the table
    for _ in range(2):                                         # the first round warms the GPU up
        base = timed(lambda: run(lib, table, q, kp, vp, out))
        base_read = timed(lambda: read(lib, table, kp, vp, ko, vo))
    print(f"{SEQS} sequences x {S} cached tokens, {moved / 2**20:.0f} MiB of keys and values per step")
    print(f"contiguous, no table: attention {base * 1e3:.2f} ms ({moved / base / 1e9:.0f} GB/s read), max error "
          f"vs PyTorch {(out - ref).abs().max():.1e}; copying alone {base_read * 1e3:.2f} ms "
          f"({2 * moved / base_read / 1e9:.0f} GB/s read and written)")
    print("        attention, time vs contiguous     copying alone                max error")
    print("block   in order   scattered               in order   scattered        vs PyTorch")
    for block in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        att, rd, err = [], [], 0
        for scattered in (False, True):
            lib, table = setup(block, scattered, gen)
            kp, vp = pools(K, V, table, block)
            att.append(timed(lambda: run(lib, table, q, kp, vp, out)) / base)
            err = max(err, (out - ref).abs().max().item())
            rd.append(timed(lambda: read(lib, table, kp, vp, ko, vo)) / base_read)
            assert torch.equal(ko, K) and torch.equal(vo, V)          # the copy is exact
        print(f"{block:5d}   {att[0]:7.2f}x  {att[1]:8.2f}x               {rd[0]:7.2f}x  {rd[1]:8.2f}x"
              f"           {err:.1e}")
