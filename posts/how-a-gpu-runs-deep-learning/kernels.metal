#include <metal_stdlib>
using namespace metal;

// Metal launches exactly `threads` threads, so these kernels skip bounds
// checks; CUDA rounds the grid up to whole blocks and needs `if (i < n)`.

// One thread per element: the GPU's "hello world".
kernel void vector_add(device float* out, device const float* a, device const float* b,
                       uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}

// Every thread copies one element, but reads it from position (i * stride) mod n.
// Stride 1: neighbouring threads read neighbouring addresses.
kernel void strided_copy(device float* out, device const float* a, constant uint& n,
                         constant uint& stride, uint i [[thread_position_in_grid]]) {
    out[i] = a[(i * stride) % n];
}

// Matrix multiply, one thread per output element, reading straight from memory.
kernel void matmul_naive(device const float* A, device const float* B, device float* C,
                         constant uint& n, uint2 p [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint k = 0; k < n; k++)
        acc += A[p.y * n + k] * B[k * n + p.x];
    C[p.y * n + p.x] = acc;
}

// The same, but each 16 x 16 group of threads first copies a tile of A and a
// tile of B into fast on-chip threadgroup memory, then reuses each loaded
// number 16 times. Assumes n is a multiple of 16.
kernel void matmul_tiled(device const float* A, device const float* B, device float* C,
                         constant uint& n, uint2 p [[thread_position_in_grid]],
                         uint2 t [[thread_position_in_threadgroup]]) {
    threadgroup float As[16][16];
    threadgroup float Bs[16][16];
    float acc = 0.0f;
    for (uint k0 = 0; k0 < n; k0 += 16) {
        As[t.y][t.x] = A[p.y * n + k0 + t.x];
        Bs[t.y][t.x] = B[(k0 + t.y) * n + p.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);     // wait until the tiles are loaded
        for (uint k = 0; k < 16; k++)
            acc += As[t.y][k] * Bs[k][t.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);     // wait before overwriting them
    }
    C[p.y * n + p.x] = acc;
}

// A bigger tile, 32 x 32: twice the reuse of each number read from memory.
kernel void matmul_tiled32(device const float* A, device const float* B, device float* C,
                         constant uint& n, uint2 p [[thread_position_in_grid]],
                         uint2 t [[thread_position_in_threadgroup]]) {
    threadgroup float As[32][32];
    threadgroup float Bs[32][32];
    float acc = 0.0f;
    for (uint k0 = 0; k0 < n; k0 += 32) {
        As[t.y][t.x] = A[p.y * n + k0 + t.x];
        Bs[t.y][t.x] = B[(k0 + t.y) * n + p.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);     // wait until the tiles are loaded
        for (uint k = 0; k < 32; k++)
            acc += As[t.y][k] * Bs[k][t.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);     // wait before overwriting them
    }
    C[p.y * n + p.x] = acc;
}

// One level further: each thread keeps a 4 x 4 block of outputs in registers,
// so a 16 x 16 group computes a 64 x 64 tile of C. Per step of k, a thread
// loads 4 numbers of A and 4 of B from threadgroup memory and does 16
// multiply-adds with them, instead of 2 loads per multiply-add.
// Assumes n is a multiple of 64.
kernel void matmul_registers(device const float* A, device const float* B, device float* C,
                             constant uint& n, uint2 g [[threadgroup_position_in_grid]],
                             uint2 t [[thread_position_in_threadgroup]]) {
    threadgroup float As[64][16];
    threadgroup float Bs[16][64];
    float acc[4][4] = {};
    uint row0 = g.y * 64, col0 = g.x * 64, id = t.y * 16 + t.x;
    for (uint k0 = 0; k0 < n; k0 += 16) {
        for (uint e = id; e < 1024; e += 256) {              // 256 threads load each slice's 1,024 numbers, 4 apiece
            As[e / 16][e % 16] = A[(row0 + e / 16) * n + k0 + e % 16];
            Bs[e / 64][e % 64] = B[(k0 + e / 64) * n + col0 + e % 64];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; k++) {
            float a[4], b[4];
            for (uint i = 0; i < 4; i++) a[i] = As[t.y * 4 + i][k];
            for (uint j = 0; j < 4; j++) b[j] = Bs[k][t.x * 4 + j];
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++)
                    acc[i][j] += a[i] * b[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            C[(row0 + t.y * 4 + i) * n + col0 + t.x * 4 + j] = acc[i][j];
}