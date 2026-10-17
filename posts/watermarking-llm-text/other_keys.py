import numpy as np
# Is the human-text mean below 0 a property of the scheme or of one key?
# The same windows, unique pairs only, scored with five other keys.
src = open("human.py").read()
exec(src.split("print(f\"{len(zs)} windows")[0])                     # windows, green(), z_unique() for wm.py's key
windows = []
for path in posts:
    body = re.sub(r"```.*?```|\$\$.*?\$\$|^import .*?$|<[^>]+>", " ", open(path).read().split("---", 2)[-1], flags=re.S | re.M)
    ids = tok(" ".join(body.split())).input_ids
    windows += [ids[s:s + N + 1] for s in range(0, len(ids) - N - 1, N)]
for key in (15485863, 1, 2, 3, 4, 5):
    KEY = key; _green.clear()
    z = np.array([z_unique(w) for w in windows])
    print(f"key {key:>9d}: mean z {z.mean():5.2f}, std {z.std():.2f}, above 4: {(z > 4).sum()}", flush=True)