import re
from collections import defaultdict

# lora.log and lora_extra.log have one line per run; group the seeds of each
# configuration and show the held-out blog and WikiText bits per byte at each
# checkpoint. "Fine-Tuning, and What It Destroys"'s full fine-tune (full_ft.log)
# for comparison.
runs = defaultdict(list)
for line in open("lora.log").readlines() + open("lora_extra.log").readlines():
    head, *rest = line.split(" | ")
    r, lr, seed = re.match(r"rank\s+(\d+), lr ([\d.e-]+), seed (\d)", head).groups()
    pts = {int(s): (float(b), float(w)) for s, b, w in re.findall(r"step (\d+): blog ([\d.]+), wiki ([\d.]+)", line)}
    secs = int(re.search(r"(\d+) s$", head).group(1))
    diff = float(re.search(r"max difference ([\d.e+-]+)", line).group(1))
    runs[(int(r), float(lr))].append((pts, secs, diff))
base = next(iter(runs.values()))[0][0][0]
print(f"before training: blog {base[0]:.3f}, WikiText {base[1]:.3f} bits per byte\n")
print("bits per byte after each step, seed 0 then seed 1 (rank 1 at 2e-4: one seed)")
print("rank  lr            " + "   ".join(f"{'step ' + str(s):11s}" for s in (25, 50, 100, 200, 400)).rstrip())
for (r, lr), rs in runs.items():
    for which, label in ((0, "blog"), (1, "wiki")):
        cells = [" ".join(f"{x[0][s][which]:.3f}" for x in rs) for s in (25, 50, 100, 200, 400)]
        print(f"{r:4d}  {lr:<10g}{label:4s}  " + "   ".join(f"{c:11s}" for c in cells).rstrip())
full = [{int(s): (float(b), float(w)) for s, b, w in re.findall(r"step (\d+): blog ([\d.]+), wiki ([\d.]+)", l)}
        for l in open("full_ft.log")]
for which, label in ((0, "blog"), (1, "wiki")):
    print(f"{'full, 1e-5':16s}{label:4s}  " + "   ".join(" ".join(f"{x[s][which]:.3f}" for x in full) for s in (25, 50, 100, 200, 400)))
print(f"\nseconds per 400-step run: {min(s for rs in runs.values() for _, s, _ in rs)} to "
      f"{max(s for rs in runs.values() for _, s, _ in rs)}")
print(f"merged vs unmerged, largest logit difference over all {sum(map(len, runs.values()))} LoRA runs: "
      f"{max(d for rs in runs.values() for _, _, d in rs):.1e}")
