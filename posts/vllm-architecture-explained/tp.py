# Tensor parallelism, worked for Llama-3-70B on eight H100s: each GPU holds
# an eighth of every layer, and each layer ends in two all-reduces (after
# attention's output projection and after the MLP; Megatron-LM) of the
# activations, B tokens x 8,192 numbers in 16 bits. A ring all-reduce
# sends 2(N-1)/N times the buffer from each GPU (nccl-tests). Compared with
# the time to read each GPU's share of the weights, which sets a decode
# step's floor. NVIDIA's figures: HBM 3.35 TB/s; NVLink 900 GB/s both ways
# (450 each way); PCIe Gen5 x16 64 GB/s each way.
D, L, H, KV, HD, F, V = 8192, 80, 64, 8, 128, 28672, 128256
params = L * (D * H * HD * 2 + D * KV * HD * 2 + 3 * D * F) + 2 * V * D
N, HBM = 8, 3.35e12
weights_ms = params * 2 / N / HBM * 1e3
print(f"Llama-3-70B: {params / 1e9:.1f}B parameters, {params * 2 / 1e9:.0f} GB in 16 bits; "
      f"{params * 2 / N / 1e9:.1f} GB per GPU, read in {weights_ms:.1f} ms")
print("decode batch   all-reduce bytes per GPU per step   over NVLink   over PCIe Gen5   (weights: "
      f"{weights_ms:.1f} ms)")
for B in (1, 16, 64, 256):
    sent = L * 2 * B * D * 2 * 2 * (N - 1) / N
    print(f"{B:8d}       {sent / 1e6:12.1f} MB                    {sent / 450e9 * 1e3:6.2f} ms     "
          f"{sent / 64e9 * 1e3:6.2f} ms")
