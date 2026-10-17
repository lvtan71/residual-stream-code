import numpy as np

# Each pair of dimensions rotates at its own speed; its wavelength is how many
# positions a full turn takes. Head dimension 128 (64 pairs), for two bases:
# 10,000 (the original) and 500,000 (Llama 3).
H = 128
i = np.arange(H // 2)
print(f"{'pair':>5s} " + "".join(f"{f'base {b:,}':>18s}" for b in (10_000, 500_000)))
for p in (0, 8, 16, 32, 48, 63):
    print(f"{p:5d} " + "".join(f"{2 * np.pi * b ** (p / (H // 2)):>18,.1f}" for b in (10_000, 500_000)))
for b in (10_000, 500_000):
    wl = 2 * np.pi * b ** (i / (H // 2))
    print(f"base {b:,}: {np.sum(wl > 8192):2d} of 64 pairs never finish one turn within 8,192 positions, "
          f"{np.sum(wl > 131_072):2d} within 131,072")