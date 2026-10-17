import subprocess
import sys

# "Fine-Tuning, and What It Destroys"'s full fine-tune at lr 1e-5 without
# replay, rerun with seeds 0 and 1 for the comparison rows of lora_table.py;
# seed 0 also saves its Transformer-block weights to ft_weights.pt for
# spectrum.py and truncated_update.py. Each run's summary line is kept in
# full_ft.log.
open("full_ft.log", "w").close()               # start a fresh log
for args in (["1e-5", "0", "400", "0", "ft_weights.pt"], ["1e-5", "0", "400", "1"]):
    out = subprocess.run([sys.executable, "ft.py", *args], capture_output=True, text=True)
    lines = [l for l in (out.stdout + out.stderr).splitlines() if l.startswith("lr ") or "Error" in l]
    print("\n".join(lines))
    open("full_ft.log", "a").write("".join(l + "\n" for l in lines))
