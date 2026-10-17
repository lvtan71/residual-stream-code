import subprocess
import sys

def run(args, log):
    """Run lora.py as the post did, keeping its summary line in a log."""
    out = subprocess.run([sys.executable, "lora.py", *map(str, args)], capture_output=True, text=True)
    lines = [l for l in (out.stdout + out.stderr).splitlines() if l.startswith("rank") or "Error" in l]
    print("\n".join(lines))
    with open(log, "a") as f:
        f.write("".join(l + "\n" for l in lines))

# How the post ran it: four settings with two seeds (lora.log), then, after
# the first results, three more runs (lora_extra.log). lora_table.py turns
# both logs into the post's table.
for log in ("lora.log", "lora_extra.log"):
    open(log, "w").close()                     # start fresh logs
for seed in (0, 1):
    for rank, lr in ((1, "1e-3"), (8, "1e-3"), (64, "1e-3"), (8, "2e-4")):
        run((rank, lr, seed), "lora.log")
for args in ((64, "1.25e-4", 0), (64, "1.25e-4", 1), (1, "2e-4", 0)):
    run(args, "lora_extra.log")
