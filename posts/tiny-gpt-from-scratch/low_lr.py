from sweep import train

# The model without residual connections and the baseline, both at a tenth of
# the learning rate (1e-4), two seeds each, scored on the dev slice.
BASE = dict(kind="char", steps=4000, dropout=0.2, checkpoints=[1000, 2000, 3000, 4000])
for seed in (0, 1):
    train(f"no residual, lr 1e-4 s{seed}", **BASE, seed=seed, residual=False, lr=0.0001)
for seed in (0, 1):
    train(f"baseline,    lr 1e-4 s{seed}", **BASE, seed=seed, lr=0.0001)
