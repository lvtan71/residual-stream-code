import math
# Arithmetic for an H100: a matmul kernel splits the output into 128 x 128
# tiles and hands them to 132 SMs in waves. A last, partly empty wave is paid
# for in full.
for n in (1024, 1408, 1536, 4096, 4097, 4224):
    per_side = math.ceil(n / 128)
    tiles = per_side ** 2
    waves = math.ceil(tiles / 132)
    useful = (n / (per_side * 128)) ** 2              # edge tiles are partly padding
    print(f"n = {n}: {tiles:4d} tiles -> {waves:2d} waves; SMs busy {tiles / (waves * 132):4.0%}, "
          f"useful work in the tiles {useful:4.0%}, overall {useful * tiles / (waves * 132):4.0%}")