import torch
assert torch.backends.mps.is_available(), "these Metal kernels need an Apple-silicon GPU"

# Two fused kernels for Qwen2.5-1.5B's layers, written in Metal, the Apple
# GPU's shading language (they run only on Apple silicon, through
# torch.mps.compile_shader). Each replaces several PyTorch operations with
# one pass over memory:
#   rmsnorm: y = w * half(x * rsqrt(mean(x^2) + eps)), in float, as Qwen2RMSNorm
#            computes it in eight operations
#   swiglu:  y = silu(g) * u, where g and u are the two halves of one wider
#            matmul's output, in place of a silu and a multiply
# rmsnorm takes one row per threadgroup, of THREADS threads, each reading
# VEC numbers at a time; those two numbers are what tune.py chooses.
SRC = """
#include <metal_stdlib>
using namespace metal;
constant uint THREADS = NTHREADS, VEC = NVEC;
typedef vec<half, VEC> halfv;
typedef vec<float, VEC> floatv;

kernel void rmsnorm(device const half* x, device const half* w, device half* y,
                    constant uint& n, constant float& eps,
                    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    device const halfv* xr = (device const halfv*)(x + row * n);
    float s = 0;
    for (uint i = t; i < n / VEC; i += THREADS) {          // each thread sums squares over its share
        floatv v = floatv(xr[i]);
        s += dot(v, v);
    }
    threadgroup float part[32];                           // then the threadgroup adds the shares
    s = simd_sum(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = lane < THREADS / 32 ? part[lane] : 0;
    float r = rsqrt(simd_sum(s) / n + eps);
    device halfv* yr = (device halfv*)(y + row * n);
    for (uint i = t; i < n / VEC; i += THREADS)
        yr[i] = ((device const halfv*)w)[i] * halfv(floatv(xr[i]) * r);
}

kernel void swiglu(device const half* gu, device half* y, constant uint& n,
                   uint i [[thread_position_in_grid]]) {
    uint row = i / n, j = i % n;
    float g = gu[row * 2 * n + j];
    y[i] = half(g / (1 + exp(-g))) * gu[row * 2 * n + n + j];  // rounded where PyTorch rounds
}
"""

_libs = {}

def lib(threads=256, vec=4):
    if (threads, vec) not in _libs:
        _libs[threads, vec] = torch.mps.compile_shader(
            SRC.replace("NTHREADS", str(threads)).replace("NVEC", str(vec)))
    return _libs[threads, vec]

def rmsnorm(x, w, eps, threads=256, vec=4, out=None):
    rows, n = x.numel() // x.shape[-1], x.shape[-1]
    y = torch.empty_like(x) if out is None else out
    lib(threads, vec).rmsnorm(x, w, y, n, eps, threads=rows * threads, group_size=threads)
    return y

def swiglu(gu):
    n = gu.shape[-1] // 2
    y = gu.new_empty(*gu.shape[:-1], n)
    lib().swiglu(gu, y, n, threads=y.numel())
    return y
