import subprocess
import sys

# agent.py with each model size. Each run appends its answers to
# agent_log_<size>.jsonl, which rescore.py reads.
for size in ("1.5B", "3B"):
    subprocess.run([sys.executable, "agent.py", size], check=True)
