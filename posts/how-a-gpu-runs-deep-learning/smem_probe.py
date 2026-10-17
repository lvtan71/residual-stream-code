import torch

# How much threadgroup (shared) memory will this GPU give one block?
for kb in (16, 32, 64):
    source = f"""
    #include <metal_stdlib>
    using namespace metal;
    kernel void probe(device float* out, uint i [[thread_position_in_grid]],
                      uint t [[thread_position_in_threadgroup]]) {{
        threadgroup float buf[{kb * 256}];                   // kb kilobytes of float32
        buf[t] = i;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        out[i] = buf[(t + 1) % 32] + buf[{kb * 256 - 1}];
    }}"""
    try:
        out = torch.zeros(64, device="mps")
        torch.mps.compile_shader(source).probe(out, threads=64, group_size=32)
        torch.mps.synchronize()
        print(f"{kb} KB of threadgroup memory: ok")
    except Exception as e:
        print(f"{kb} KB of threadgroup memory: refused ({str(e).split('\"')[1][:60]})")