from sweep import train

# The second round, two seeds each, on the dev slice: the chosen model, then
# one change at a time (the learning-rate schedule, then the ablations).
BASE = dict(kind="char", steps=4000, dropout=0.2, checkpoints=[1000, 2000, 3000, 4000])
for seed in (0, 1):
    train(f"baseline        s{seed}", **BASE, seed=seed)
    train(f"cosine schedule s{seed}", **BASE, seed=seed, schedule="cosine")
    train(f"no position     s{seed}", **BASE, seed=seed, position="none")
    train(f"learned pos.    s{seed}", **BASE, seed=seed, position="learned")
    train(f"post-norm       s{seed}", **BASE, seed=seed, prenorm=False)
    train(f"no residual     s{seed}", **BASE, seed=seed, residual=False)
