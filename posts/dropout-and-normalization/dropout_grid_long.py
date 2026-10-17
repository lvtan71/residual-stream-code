import subprocess
import sys

# The same grid as dropout_grid.py, four times as long: 12,000 steps of Adam.
subprocess.run([sys.executable, "dropout_grid.py", "12000"], check=True)
