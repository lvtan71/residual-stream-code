import re
from collections import defaultdict

# distill.log and teacher.log, as one table: bits per byte at each checkpoint,
# seed 0 then seed 1.
runs = defaultdict(list)
for line in open("distill.log"):
    if not re.match(r"^(hard|soft|mix)", line):
        continue
    mode = line.split()[0]
    runs[mode].append({int(s): (float(b), float(w)) for s, b, w in re.findall(r"step (\d+): blog ([\d.]+), wiki ([\d.]+)", line)})
print(open("teacher.log").read().strip())
print(f"student before training: blog {runs['hard'][0][0][0]:.3f}, wiki {runs['hard'][0][0][1]:.3f} bits per byte\n")
print("student, bits per byte after each step, seed 0 then seed 1")
# The header's leading spaces were widened after the run (6 to 10) to line up
# with the rows; the post and three_labels.out show the widened header.
print("labels          " + "   ".join(f"{'step ' + str(s):11s}" for s in (50, 100, 200, 400)).rstrip())
for mode, label in (("hard", "hard"), ("soft", "soft"), ("mix", "half each")):
    for which, name in ((0, "blog"), (1, "wiki")):
        print(f"{label:9s} {name}  " + "   ".join(" ".join(f"{r[s][which]:.3f}" for r in runs[mode]) for s in (50, 100, 200, 400)))
