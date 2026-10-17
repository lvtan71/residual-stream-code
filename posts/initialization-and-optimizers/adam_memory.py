# Bytes of training state per parameter, mixed-precision training (ZeRO's accounting).
params = 8e9                                   # an 8B model
state = {"BF16 weights": 2, "BF16 gradients": 2,
         "FP32 master weights": 4, "Adam m (FP32)": 4, "Adam v (FP32)": 4}
for name, b in state.items():
    print(f"{name:20s} {b:2d} bytes/param  {b * params / 1e9:5.0f} GB")
total = sum(state.values())
print(f"{'total':20s} {total:2d} bytes/param  {total * params / 1e9:5.0f} GB, "
      f"{total * params / 80e9:.1f} H100s of memory before a single activation")
