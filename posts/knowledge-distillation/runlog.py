import subprocess
import sys

# How the post ran the training scripts: as separate processes, keeping each
# run's summary line in a log that the tables are made from. The lines also go
# to stderr, to follow progress.
def run(script, *args, log, keep):
    out = subprocess.run([sys.executable, script, *map(str, args)], capture_output=True, text=True)
    lines = [l for l in (out.stdout + out.stderr).splitlines() if l.startswith(keep) or "Error" in l]
    print("\n".join(lines), file=sys.stderr, flush=True)
    with open(log, "a") as f:
        f.write("".join(l + "\n" for l in lines))
