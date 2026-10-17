import os
import subprocess
import sys
from runlog import run

# The teacher's top 64 on each seed's batches, then the student trained three
# ways (hard, soft, mix), two seeds each, at lr 1e-5, then the table.
for log in ("teacher.log", "distill.log"):
    if os.path.exists(log):
        os.remove(log)
for seed in (0, 1):
    run("teacher.py", seed, log="teacher.log", keep="teacher")
for seed in (0, 1):
    for mode in ("hard", "soft", "mix"):
        run("distill.py", mode, "1e-5", seed, log="distill.log", keep=("hard", "soft", "mix"))
subprocess.run([sys.executable, "distill_table.py"], check=True)
