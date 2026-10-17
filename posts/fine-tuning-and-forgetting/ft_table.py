import re
from collections import defaultdict

# ft.out as one table: bits per byte at each checkpoint for every setting,
# seed 0 then seed 1, on the held-out blog, WikiText-2 and Vietnamese Wikipedia.
runs = defaultdict(list)
for line in open("ft.out"):
    m = re.match(r"lr ([\d.e-]+), replay (\d+)%", line)
    if m:
        runs[(float(m.group(1)), int(m.group(2)))].append(
            {int(s): (float(b), float(w), float(v)) for s, b, w, v in
             re.findall(r"step (\d+): blog ([\d.]+), wiki ([\d.]+), vi ([\d.]+)", line)})
first = next(iter(runs.values()))[0][0]
print(f"before training: blog {first[0]:.3f}, WikiText {first[1]:.3f}, Vietnamese {first[2]:.3f} bits per byte\n")
print("bits per byte after each step, seed 0 then seed 1")
print(f"{'lr      replay':23s}" + "   ".join(f"{'step ' + str(s):11s}" for s in (25, 50, 100, 200, 400)).rstrip())
for (lr, replay), rs in sorted(runs.items()):
    for which, name in ((0, "blog"), (1, "wiki"), (2, "vi")):
        print(f"{lr:<8g}{replay:3d}%  {name:5s}    " +
              "   ".join(" ".join(f"{r[s][which]:.3f}" for r in rs) for s in (25, 50, 100, 200, 400)))
