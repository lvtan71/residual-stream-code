import sys
from train import run, full_train, held_out

# The same model and schedule, trained on the first quarter or half of the
# training text, scored on the held-out posts: how much does more data help?
frac, seed = float(sys.argv[1]), int(sys.argv[2])
split = {"fit": full_train[:int(frac * len(full_train))], "score": held_out}
_, curve, passes = run("char", seed, split, 4000, checkpoints=(250, 500, 750, 1000, 2000, 3000), dropout=0.2, schedule="cosine")
print(f"{frac:.0%} of the training text ({len(split['fit']):,} characters), seed {seed}: held-out bits per byte at "
      + " / ".join(f"{s:,}: {v:.3f}" for s, v in curve.items()) + f" steps ({passes:.0f} passes)")
