import os
import subprocess
import sys
from runlog import run

# The teacher's top 64 on the held-out blog (teacher_val.py), then seed 0
# again with the distance to the teacher logged, and the table.
if os.path.exists("distill_trace.log"):
    os.remove("distill_trace.log")
subprocess.run([sys.executable, "teacher_val.py"], check=True)
for mode in ("soft", "hard", "mix"):
    run("distill.py", mode, "1e-5", 0, "trace", log="distill_trace.log", keep=("hard", "soft", "mix"))
subprocess.run([sys.executable, "trace_table.py"], check=True)
