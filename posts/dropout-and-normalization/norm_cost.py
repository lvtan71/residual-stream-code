# One normalization over a training batch of 8 x 8,192 tokens at width 4,096,
# in BF16 on an H100 SXM, against one of the layer's matmuls.
tokens, D, byte = 8 * 8192, 4096, 2
vector_flops = 67e12        # FP32 on the ordinary cores, where a norm's math runs
tensor_flops = 989e12       # BF16 on the tensor cores: matmuls only
bandwidth = 3.35e12
x_bytes = tokens * D * byte                           # one full read (or write) of the activations
for name, flops_per_elem, passes_unfused in [("LayerNorm", 8, 4), ("RMSNorm", 5, 3)]:
    t_math = tokens * D * flops_per_elem / vector_flops
    print(f"{name:9s} math {t_math * 1e6:5.0f} us   memory: unfused {passes_unfused} passes "
          f"{passes_unfused * x_bytes / bandwidth * 1e3:.2f} ms, fused 2 passes {2 * x_bytes / bandwidth * 1e3:.2f} ms")
matmul = 2 * tokens * D * D / tensor_flops
print(f"one {D} x {D} matmul on the same tokens, at tensor-core peak: {matmul * 1e3:.2f} ms")
