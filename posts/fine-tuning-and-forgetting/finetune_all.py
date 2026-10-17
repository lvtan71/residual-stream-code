import subprocess
import sys

# How the post ran it: every setting with two seeds, each run's summary line
# kept in ft.out, which ft_table.py turns into the post's table.
open("ft.out", "w").close()                    # start a fresh ft.out
for seed in (0, 1):
    for lr, replay in (("1e-5", "0"), ("1e-4", "0"), ("1e-4", "0.25"), ("1e-5", "0.25")):
        out = subprocess.run([sys.executable, "ft.py", lr, replay, "400", str(seed)], capture_output=True, text=True)
        lines = [l for l in (out.stdout + out.stderr).splitlines() if l.startswith("lr ") or "Error" in l]
        print("\n".join(lines))
        open("ft.out", "a").write("".join(l + "\n" for l in lines))
